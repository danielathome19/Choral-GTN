import os
import mido
import pydub
import scipy
import librosa
import seaborn
import numpy as np
import pandas as pd
import skimage.measure
import matplotlib.pyplot as plt
from io import BytesIO
from midi2audio import FluidSynth
from scipy.spatial import distance
from skimage.transform import resize
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist, squareform


window_size = 2048  # (samples/frame)
hop_length = 1024  # overlap 50% (samples/frame)
sr_desired = 44100
p = 2  # pooling factor
p2 = 3  # 2pool3
L_sec_near = 14  # lag near context in seconds
L_near = round(L_sec_near * sr_desired / hop_length)  # conversion of lag L seconds to frames


def compute_ssm(X, metric=cosine):
    """Computes the self-similarity matrix of X."""
    D = pdist(X, metric=metric)
    D = squareform(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if np.isnan(D[i, j]):
                D[i, j] = 0
    D /= D.max()
    return 1 - D


def mel_spectrogram(sr_desired, filepath, window_size, hop_length):
    """Calculates the mel spectrogram in dB"""
    y, sr = librosa.load(filepath, sr=None)
    if sr != sr_desired:
        y = librosa.core.resample(y, orig_sr=sr, target_sr=sr_desired)
        sr = sr_desired
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size, hop_length=hop_length, n_mels=80, fmin=80,
                                       fmax=16000)
    S_to_dB = librosa.power_to_db(S, ref=np.max)  # convert S in dB
    return S_to_dB  # S_to_dB is the spectrogam in dB


def fourier_transform(sr_desired, name_song, window_size, hop_length):
    """This function calculates the mel spectrogram in dB with Librosa library"""
    y, sr = librosa.load(name_song, sr=None)
    if sr != sr_desired:
        y = librosa.core.resample(y, orig_sr=sr, target_sr=sr_desired)
        sr = sr_desired
    stft = np.abs(librosa.stft(y=y, n_fft=window_size, hop_length=hop_length))
    return stft


def max_pooling(stft, pooling_factor):
    x_prime = skimage.measure.block_reduce(stft, block_size=pooling_factor, func=np.max)
    return x_prime


def sslm_gen(spectrogram, pooling_factor, lag, mode, feature):
    padding_factor = lag
    pad = np.full((spectrogram.shape[0], padding_factor), -70)  # 80x30 frame matrix of -70dB corresponding to padding
    S_padded = np.concatenate((pad, spectrogram), axis=1)  # padding 30 frames with noise at -70dB at the beginning
    x_prime = max_pooling(S_padded, pooling_factor)
    x = []
    if feature == "mfcc":
        # MFCCs calculation from DCT-Type II
        MFCCs = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
        MFCCs = MFCCs[1:, :]  # 0 componen ommited
        # Bagging frames
        m = 2  # bagging parameter in frames
        x = [np.roll(MFCCs, n, axis=1) for n in range(m)]
    elif feature == "chroma":
        PCPs = librosa.feature.chroma_stft(S=x_prime, sr=sr_desired, n_fft=window_size, hop_length=hop_length)
        PCPs = PCPs[1:, :]
        m = 2
        x = [np.roll(PCPs, n, axis=1) for n in range(m)]

    x_hat = np.concatenate(x, axis=0)

    # Cosine distance calculation: D[N/p,L/p] matrix
    distances = np.zeros((x_hat.shape[1], padding_factor // p))  # D has as dimensions N/p and L/p
    for i in range(x_hat.shape[1]):  # iteration in columns of x_hat
        for l in range(padding_factor // p):
            if i - (l + 1) < 0:
                cur_dist = 1
            elif i - (l + 1) < padding_factor // p:
                cur_dist = 1
            else:
                cur_dist = 0
                if mode == "cos":
                    cur_dist = distance.cosine(x_hat[:, i],
                                               x_hat[:, i - (l + 1)])  # cosine distance between columns i and i-L
                elif mode == "euc":
                    cur_dist = distance.euclidean(x_hat[:, i],
                                                  x_hat[:, i - (l + 1)])  # euclidian distance between columns i and i-L
                if cur_dist == float('nan'):
                    cur_dist = 0
            distances[i, l] = cur_dist

    # Threshold epsilon[N/p,L/p] calculation
    kappa = 0.1
    epsilon = np.zeros((distances.shape[0], padding_factor // p))  # D has as dimensions N/p and L/p
    for i in range(padding_factor // p, distances.shape[0]):  # iteration in columns of x_hat
        for l in range(padding_factor // p):
            epsilon[i, l] = np.quantile(np.concatenate((distances[i - l, :], distances[i, :])), kappa)

    # We remove the padding done before
    distances = distances[padding_factor // p:, :]
    epsilon = epsilon[padding_factor // p:, :]
    x_prime = x_prime[:, padding_factor // p:]

    # Self Similarity Lag Matrix
    sslm = scipy.special.expit(1 - distances / epsilon)  # aplicación de la sigmoide
    sslm = np.transpose(sslm)
    sslm = skimage.measure.block_reduce(sslm, block_size=3, func=np.max)

    # Check if SSLM has nans and if it has them, substitute them by 0
    for i in range(sslm.shape[0]):
        for j in range(sslm.shape[1]):
            if np.isnan(sslm[i, j]):
                sslm[i, j] = 0

    # if mode == "euc":
    #     return sslm, x_prime

    # return sslm
    return sslm, x_prime


def ssm_gen(spectrogram, pooling_factor):
    x_prime = max_pooling(spectrogram, pooling_factor)
    # MFCCs calculation from DCT-Type II
    MFCCs = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
    MFCCs = MFCCs[1:, :]  # 0 componen ommited

    # Bagging frames
    m = 2  # bagging parameter in frames
    x = [np.roll(MFCCs, n, axis=1) for n in range(m)]
    x_hat = np.concatenate(x, axis=0)
    x_hat = np.transpose(x_hat)

    ssm = compute_ssm(x_hat)

    # Check if SSLM has nans and if it has them, substitute them by 0
    for i in range(ssm.shape[0]):
        for j in range(ssm.shape[1]):
            if np.isnan(ssm[i, j]):
                ssm[i, j] = 0

    return ssm


def midi_to_wav(midi_filepath):
    """Converts a MIDI file to a WAV file."""
    fs = FluidSynth()
    wav_filepath = midi_filepath[:midi_filepath.rindex('.')] + '.wav'
    fs.midi_to_audio(midi_filepath, wav_filepath)
    return wav_filepath


def util_main_helper(feature, mid_filepath, mode="cos", predict=False, savename="", display=False):
    filepath = midi_to_wav(mid_filepath)
    print(filepath)
    try:
        sslm_near = None
        if feature == "mfcc":
            mel = mel_spectrogram(sr_desired, filepath, window_size, hop_length)
            if mode == "cos":
                sslm_near = sslm_gen(mel, p, L_near, mode=mode, feature="mfcc")[0]
            elif mode == "euc":
                sslm_near = sslm_gen(mel, p, L_near, mode=mode, feature="mfcc")[0]
                if sslm_near.shape[1] < max_pooling(mel, 6).shape[1]:
                    sslm_near = np.hstack((np.ones((301, 1)), sslm_near))
                elif sslm_near.shape[1] > max_pooling(mel, 6).shape[1]:
                    sslm_near = sslm_near[:, 1:]
        elif feature == "chroma":
            stft = fourier_transform(sr_desired, filepath, window_size, hop_length)
            sslm_near = sslm_gen(stft, p, L_near, mode=mode, feature="chroma")[0]
            if mode == "euc":
                if sslm_near.shape[1] < max_pooling(stft, 6).shape[1]:
                    sslm_near = np.hstack((np.ones((301, 1)), sslm_near))
                elif sslm_near.shape[1] > max_pooling(stft, 6).shape[1]:
                    sslm_near = sslm_near[:, 1:]
        elif feature == "mls":
            mel = mel_spectrogram(sr_desired, filepath, window_size, hop_length)
            sslm_near = ssm_gen(mel, pooling_factor=6)
    except Exception as e:
        os.remove(filepath)
        raise e
    os.remove(filepath)

    if display:
        # recurrence = librosa.segment.recurrence_matrix(sslm_near, mode='affinity', k=sslm_near.shape[1])
        plt.figure(figsize=(15, 10))
        if sslm_near is not None:
            if feature == "mls":
                plt.title("Mel Log-scaled Spectrogram - Self-Similarity matrix (MLS SSM)")
                plt.imshow(sslm_near, origin='lower', cmap='plasma', aspect=0.8)  # switch to recurrence if desired
            else:
                plt_title = "Self-Similarity Lag Matrix (SSLM): "
                if feature == "chroma":
                    plt_title += "Chromas, "
                else:
                    plt_title += "MFCCs, "
                if mode == "cos":
                    plt_title += "Cosine Distance"
                else:
                    plt_title += "Euclidian Distance"
                plt.title(plt_title)
                plt.imshow(sslm_near.astype(np.float32), origin='lower', cmap='viridis', aspect=0.8)
                # switch to recurrence if desired
            plt.show()
    if not predict and sslm_near is not None:
        # Save matrices and sslms as numpy arrays in separate paths
        np.save(filepath, sslm_near)
    else:
        return sslm_near


data_path = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Soprano\Isolated\2_WeepS.mid")
util_main_helper("mls", data_path, mode="cos", predict=True)
