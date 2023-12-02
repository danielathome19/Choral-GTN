import gc
import sys
import ast
import time
import random
import logging
import warnings
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from main import *
from Transformer import *
from keras import layers
from keras import losses
from functools import partial
from keras import backend as k
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import AdamW
from keras.models import Sequential
from keras_tuner import RandomSearch
from keras.src.utils import plot_model
from keras.callbacks import EarlyStopping
from data_utils import key_signature_to_number
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras_tuner import HyperParameters, Objective, tuners


def plot_histories(model, feature1, feature2, title, ylabel, filename=None):
    plt.plot(model.history.history[feature1])
    plt.plot(model.history.history[feature2])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    if filename is not None:
        plt.savefig(filename)


def generate_intro(dataset="Soprano", generate_len=50, temperature=0.5, key=None,
                   time_sig=None, tempo: int = None, entrance: float = None):
    with open(f"Weights/Composition_Intro/{dataset}_notes_vocab.pkl", "rb") as f:
        notes_vocab = pkl.load(f)
    with open(f"Weights/Composition_Intro/{dataset}_durations_vocab.pkl", "rb") as f:
        durations_vocab = pkl.load(f)

    key, time_sig, tempo, entrance = validate_and_generate_metatrack(dataset, key, time_sig, tempo, entrance)
    start_tokens = [f"START", key, time_sig, tempo, "rest"]
    start_durations = ["0.0", "0.0", "0.0", "0.0", str(entrance)]
    print(f"Generating {dataset} intro with key={key}, time_sig={time_sig}, tempo={tempo}, entrance={entrance}...")
    while True:
        try:
            model = build_model(len(notes_vocab), len(durations_vocab),
                                feed_forward_dim=512, num_heads=8, verbose=False)
            model.load_weights(f"Weights/Composition_Intro/{dataset}/checkpoint.ckpt")
            music_generator = MusicGenerator(notes_vocab, durations_vocab, generate_len=generate_len)
            clef = "treble" if dataset in ["Soprano", "Alto"] else dataset.lower()
            info = music_generator.generate(start_tokens, start_durations, max_tokens=generate_len, clef=clef,
                                            temperature=temperature, model=model, intro=True, instrument=dataset)
            midi_stream = info[-1]["midi"]
            timestr = time.strftime("%Y%m%d-%H%M%S")
            output_dir = f"Data/Generated/Intro_{dataset}"
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            filename = os.path.join(output_dir, "output-" + timestr + ".mid")
            midi_stream.write("midi", fp=filename)
            # Check the output MIDI file -- if it's less than 250 bytes, it's probably junk; retry
            if os.path.getsize(filename) < 250:
                os.remove(filename)
                # print(f"Failed to generate intro for {dataset}; retrying...")
            else:
                break
        except Exception as e:
            print(e)
    print(f"Generated intro for {dataset}\n")
    return key, time_sig, tempo, entrance, filename


def train_intro_model(dataset="Soprano", epochs=100):
    """Trains a Transformer model to generate the first 8 measures of the specified voice part."""
    PARSED_DATA_PATH = f"Data/Glob/Combined_mm1-8/Combined_{dataset}_choral_"
    INCLUDE_AUGMENTED = True
    BATCH_SIZE = 128
    DATASET_REPETITIONS = 1
    LOAD_MODEL = True
    GENERATE_LEN = 50

    # Load the parsed data
    notes, durations = load_parsed_files(PARSED_DATA_PATH)
    if INCLUDE_AUGMENTED:
        for i in range(1, 5):
            aug_path = f"Data/Glob/Combined_mm1-8/Combined_aug{i}_{dataset}_choral_"
            aug_notes, aug_durations = load_parsed_files(aug_path)
            notes += aug_notes
            durations += aug_durations

    # For every string in the notes and durations lists, remove all instances of "{dataset}:"
    notes = [note.replace(f"{dataset}:", "") for note in notes]
    durations = [duration.replace(f"{dataset}:", "") for duration in durations]

    notes_seq_ds, notes_vectorize_layer, notes_vocab = create_transformer_dataset(notes, BATCH_SIZE)
    durations_seq_ds, durations_vectorize_layer, durations_vocab = create_transformer_dataset(durations, BATCH_SIZE)
    seq_ds = tf.data.Dataset.zip((notes_seq_ds, durations_seq_ds))

    notes_vocab_size = len(notes_vocab)
    durations_vocab_size = len(durations_vocab)

    # Save vocabularies
    if not os.path.exists(f"Weights/Composition_Intro"):
        os.mkdir(f"Weights/Composition_Intro")
    with open(f"Weights/Composition_Intro/{dataset}_notes_vocab.pkl", "wb") as f:
        pkl.dump(notes_vocab, f)
    with open(f"Weights/Composition_Intro/{dataset}_durations_vocab.pkl", "wb") as f:
        pkl.dump(durations_vocab, f)

    # Create the training set of sequences and the same sequences shifted by one note
    def prepare_inputs(notes, durations):
        notes = tf.expand_dims(notes, -1)
        durations = tf.expand_dims(durations, -1)
        tokenized_notes = notes_vectorize_layer(notes)
        tokenized_durations = durations_vectorize_layer(durations)
        x = (tokenized_notes[:, :-1], tokenized_durations[:, :-1])
        y = (tokenized_notes[:, 1:], tokenized_durations[:, 1:])
        return x, y

    ds = seq_ds.map(prepare_inputs).repeat(DATASET_REPETITIONS)

    gc.collect()
    model = build_model(notes_vocab_size, durations_vocab_size, feed_forward_dim=512, num_heads=8)
    plot_model(model, to_file=f'Images/{dataset}_intro_model.png',
               show_shapes=True, show_layer_names=True, expand_nested=True)

    if LOAD_MODEL:
        model.load_weights(f"Weights/Composition_Intro/{dataset}/checkpoint.ckpt")
        print("Loaded model weights")

    class ClearGarbage(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            k.clear_session()

    checkpoint_callback = callbacks.ModelCheckpoint(filepath=f"Weights/Composition_Intro/{dataset}/checkpoint.ckpt",
                                                    save_weights_only=True, save_freq="epoch", verbose=0)
    tensorboard_callback = callbacks.TensorBoard(log_dir=f"Logs/Intro_{dataset}")

    # Tokenize starting prompt
    music_generator = MusicGenerator(notes_vocab, durations_vocab, generate_len=GENERATE_LEN)
    model.fit(ds, epochs=epochs, callbacks=[checkpoint_callback, tensorboard_callback, music_generator, ClearGarbage()])
    model.save(f"Weights/Composition_Intro/{dataset}.keras")

    # Test the model
    info = music_generator.generate(["START"], ["0.0"], max_tokens=50, temperature=0.5)
    midi_stream = info[-1]["midi"]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    midi_stream.write("midi", fp=os.path.join(f"Data/Generated/Intro_{dataset}", "output-" + timestr + ".mid"))


def train_markov_composition_model():
    import markovify
    GENERATE_LEN = 3
    INCLUDE_AUGMENTED = False
    DATAPATH = "Data/Glob/Combined_choral"

    voices = ["S", "A", "T", "B"]
    voice_parts_notes = {}
    voice_parts_durations = {}
    for voice in voices:
        voice_parts_notes[voice] = load_pickle_from_slices(f"{DATAPATH}/Combined_{voice}_choral_notes", False)
        voice_parts_durations[voice] = load_pickle_from_slices(f"{DATAPATH}/Combined_{voice}_choral_durations",
                                                               False)
        if INCLUDE_AUGMENTED:
            for i in range(1, 5):
                aug_notes = load_pickle_from_slices(f"{DATAPATH}/Combined_aug{i}_{voice}_choral_notes", False)
                aug_dur = load_pickle_from_slices(f"{DATAPATH}/Combined_aug{i}_{voice}_choral_durations", False)
                voice_parts_notes[voice] += aug_notes
                voice_parts_durations[voice] += aug_dur

    def train_markov_models(voice_parts_notes, voice_parts_durations):
        voice_models = {}
        # Train separate Markov models for notes and durations for each voice part
        for voice in voice_parts_notes:
            print("Training Markov models for", voice)
            notes_model = markovify.NewlineText('\n'.join(voice_parts_notes[voice]))
            durations_model = markovify.NewlineText('\n'.join(voice_parts_durations[voice]))
            voice_models[voice] = {'notes': notes_model, 'durations': durations_model}
        return voice_models

    def generate_choral_sequence(voice_models, length=25):
        # Generate a sequence for each voice part using their respective models
        generated_sequence = {'notes': [], 'durations': []}
        print("Generating choral sequence...")
        for _ in range(length):
            for voice in voice_models:
                for _attempt in range(100):
                    sentence_n = voice_models[voice]['notes'].make_sentence()
                    sentence_d = voice_models[voice]['durations'].make_sentence()
                    if sentence_n is not None and sentence_d is not None:
                        generated_sequence['notes'].append(sentence_n)
                        generated_sequence['durations'].append(sentence_d)
                        break
        return generated_sequence

    def generate_midi(generated_sequence):
        print("Generating MIDI file...")
        voice_streams = {
            'S': music21.stream.Part(),
            'A': music21.stream.Part(),
            'T': music21.stream.Part(),
            'B': music21.stream.Part()
        }

        clefs = {
            'S': music21.clef.TrebleClef(),
            'A': music21.clef.TrebleClef(),
            'T': music21.clef.Treble8vbClef(),
            'B': music21.clef.BassClef()
        }

        for voice, stream in voice_streams.items():
            stream.append(clefs[voice])

        start_notes = ["S:START", "A:START", "T:START", "B:START"]
        start_durations = ["0.0", "0.0", "0.0", "0.0"]
        for sample_token, sample_duration in zip(start_notes, start_durations):
            voice_type = sample_token.split(":")[0]
            new_note = get_choral_midi_note(sample_token, sample_duration)
            if new_note is not None:
                if voice_type not in ["S", "A", "T", "B"]:
                    voice_streams["S"].append(new_note)
                else:
                    voice_streams[voice_type].append(new_note)

        intro = True

        all_notes = []
        all_durations = []
        for sentence in generated_sequence['notes']:
            if sentence is not None:
                all_notes += sentence.split(" ")
        for sentence in generated_sequence['durations']:
            if sentence is not None:
                all_durations += sentence.split(" ")
        if len(all_notes) != len(all_durations):
            if len(all_notes) > len(all_durations):
                all_durations += "0.0" * (len(all_notes) - len(all_durations))
            else:
                all_notes += "S:rest" * (len(all_durations) - len(all_notes))

        for sample_note, sample_duration in zip(all_notes, all_durations):
            voice_type = sample_note.split(":")[0]
            new_note = get_choral_midi_note(sample_note, sample_duration)

            if (isinstance(new_note, music21.chord.Chord) or isinstance(new_note, music21.note.Note) or
                isinstance(new_note, music21.note.Rest)) and sample_duration == "0.0":
                continue
            elif (isinstance(new_note, music21.tempo.MetronomeMark) or
                  isinstance(new_note, music21.key.Key) or
                  isinstance(new_note, music21.meter.TimeSignature)):
                if intro:
                    intro = False
                else:
                    continue

            if new_note is not None:
                if voice_type not in ["S", "A", "T", "B"]:
                    voice_streams["S"].append(new_note)
                else:
                    voice_streams[voice_type].append(new_note)

            if "START" in sample_note:
                continue

        midi_stream = music21.stream.Score()
        for voice, stream in voice_streams.items():
            midi_stream.insert(0, stream)
        return midi_stream

    if not os.path.exists(f"Weights/MarkovChain"):
        os.makedirs(f"Weights/MarkovChain")
        voice_models = train_markov_models(voice_parts_notes, voice_parts_durations)
        for voice in voice_models:
            with open(f"Weights/MarkovChain/{voice}_notes_model.pkl", "wb") as f:
                pkl.dump(voice_models[voice]['notes'], f)
            with open(f"Weights/MarkovChain/{voice}_durations_model.pkl", "wb") as f:
                pkl.dump(voice_models[voice]['durations'], f)
        print("Saved Markov models")
    else:
        voice_models = {}
        for voice in voices:
            with open(f"Weights/MarkovChain/{voice}_notes_model.pkl", "rb") as f:
                voice_models[voice] = {'notes': pkl.load(f)}
            with open(f"Weights/MarkovChain/{voice}_durations_model.pkl", "rb") as f:
                voice_models[voice]['durations'] = pkl.load(f)
        print("Loaded Markov models")

    for i in range(10):
        generated_sequence = generate_choral_sequence(voice_models, GENERATE_LEN)
        print(generated_sequence)
        midi_stream = generate_midi(generated_sequence)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(f"Data/Generated/MarkovChain"):
            os.makedirs(f"Data/Generated/MarkovChain")
        midi_stream.write("midi", fp=os.path.join(f"Data/Generated/MarkovChain", "output-" + timestr + ".mid"))
    pass


# region FeatureModels

def train_duration_model(dataset="Soprano", epochs=100):
    """Trains a Bi-LSTM model to predict the duration of the next note given the previous note and duration."""
    df = pd.read_csv(f"Data/Tabular/{dataset}.csv", sep=';')
    df = df[['event', 'time']]

    # Normalize the data
    scaler = StandardScaler()
    df['event'] = df['event'].apply(ast.literal_eval).apply(np.array)
    df['time'] = df['time'].apply(ast.literal_eval).apply(np.array)
    df = df[df['time'].apply(len) > 0]  # Remove empty sequences
    df = df[df['event'].apply(len) > 0]
    df['time'] = df['time'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten())

    # The model will take in the event and the previous time as input and predict the next time
    df['time_prev'] = df['time'].apply(lambda x: np.concatenate([[0], x[:-1]]))
    df['time_next'] = df['time'].apply(lambda x: np.concatenate([x[1:], [0]]))
    inputs = np.array(df[['event', 'time_prev']])
    outputs = np.array(df['time_next'])

    # Find the longest event array (index 0) and longest time array (index 1)
    # Pad all other arrays to this length using -1 (rest) for event and 0.0 for time
    max_event_len = max([len(x[0]) for x in inputs])
    max_time_len = max([len(x[1]) for x in inputs])
    max_output_len = max([len(x) for x in outputs])
    inputs_e = np.array([np.concatenate([x, np.full(max_event_len-len(x), -1)]).astype(int)
                         for x in np.array(df['event'])])
    inputs_t = np.array([np.concatenate([x, np.full(max_time_len-len(x), 0.)]).astype(float)
                         for x in np.array(df['time_prev'])])
    inputs = np.stack((inputs_e, inputs_t), axis=-1)
    outputs = np.array([np.concatenate([x, np.full(max_output_len-len(x), 0.)]).astype(float) for x in outputs])

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Bi-Directonal LSTM
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(max_event_len, 2)))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.TimeDistributed(layers.Dense(64, activation='relu')))
    model.add(layers.TimeDistributed(layers.Dense(1, activation='linear')))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.summary()
    plot_model(model, to_file=f'Images/{dataset}_duration_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', f"{dataset} Duration Model Loss (MSE)", 'Loss (MSE)')
    plot_histories(model, 'mae', 'val_mae', f"{dataset} Duration Model MAE", 'MAE')

    # Save the model, scaler, and max length (all 3 max lengths should be the same)
    model.save(f"Weights/Duration/{dataset}_model.h5")
    pkl.dump(scaler, open(f"Weights/Duration/{dataset}_time_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights/Duration/{dataset}_seq_len.pkl", 'wb'))

    # Test the model with the first event array; pad the arrays to match the max lengths and then trim after prediction
    event = np.concatenate([df['event'][0], np.full(max_event_len - len(df['event'][0]), -1)]).astype(int)
    time_prev = np.concatenate([df['time_prev'][0], np.full(max_time_len - len(df['time_prev'][0]), 0.)]).astype(float)
    time_next = df['time_next'][0]
    # Reshape the input to match the model input layer
    input_data = np.array([event, time_prev]).reshape((1, max_event_len, 2))
    time_pred = model.predict(input_data)
    # Rescale the time to match the original data
    time_next = scaler.inverse_transform(time_next.reshape(-1, 1)).flatten()
    time_pred = time_pred.squeeze()  # Remove dimensions of size 1
    time_pred = time_pred[:len(time_next)]  # Trim the predicted time to match the actual time
    time_pred = scaler.inverse_transform(time_pred.reshape(-1, 1)).flatten()
    print("Actual time: ", time_next)
    print("Predicted time: ", time_pred)
    print("Difference: ", time_next - time_pred)

    return model, scaler, max_event_len


def train_tempo_model(epochs=50):
    """Trains an LSTM model to predict the tempo of a piece based on the events and event times."""
    df = pd.read_csv(f"Data/Tabular/Soprano.csv", sep=';')
    df = df[['event', 'time', 'tempo']]

    # Normalize the data
    time_scaler = StandardScaler()
    tempo_scaler = StandardScaler()
    df['event'] = df['event'].apply(ast.literal_eval).apply(np.array)
    df['time'] = df['time'].apply(ast.literal_eval).apply(np.array)
    df['tempo'] = df['tempo'].apply(ast.literal_eval).apply(np.array)
    df = df[df['time'].apply(len) > 0]
    df = df[df['event'].apply(len) > 0]
    df = df[df['tempo'].apply(len) > 0]
    df['time'] = df['time'].apply(lambda x: time_scaler.fit_transform(x.reshape(-1, 1)).flatten())
    df['tempo'] = df['tempo'].apply(lambda x: tempo_scaler.fit_transform(x.reshape(-1, 1)).flatten())
    inputs = np.array(df[['event', 'time']])
    outputs = np.array(df['tempo'])

    # Pad the inputs and outputs to the max length
    max_event_len = max([len(x[0]) for x in inputs])
    max_time_len = max([len(x[1]) for x in inputs])
    max_output_len = max([len(x) for x in outputs])
    inputs_e = np.array([np.concatenate([x, np.full(max_event_len-len(x), -1)]).astype(int)
                         for x in np.array(df['event'])])
    inputs_t = np.array([np.concatenate([x, np.full(max_time_len-len(x), 0.)]).astype(float)
                         for x in np.array(df['time'])])
    inputs = np.stack((inputs_e, inputs_t), axis=-1)
    outputs = np.array([np.concatenate([x, np.full(max_output_len-len(x), 0.)]).astype(float) for x in outputs])

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # LSTM
    model = Sequential()
    model.add(layers.LSTM(64, activation='tanh', input_shape=(max_event_len, 2), return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(64, activation='tanh')))
    model.add(layers.TimeDistributed(layers.Dense(1, activation='linear')))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    plot_model(model, to_file=f'Images/tempo_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', "Tempo Model Loss (MSE)", 'Loss')
    plot_histories(model, 'mae', 'val_mae', "Tempo Model MAE", 'MAE')

    # Save the model, scalers, and max length
    model.save(f"Weights/Tempo/model.h5")
    pkl.dump(time_scaler, open(f"Weights/Tempo/time_scaler.pkl", 'wb'))
    pkl.dump(tempo_scaler, open(f"Weights/Tempo/tempo_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights/Tempo/seq_len.pkl", 'wb'))

    # Test the model with the first event array; pad the arrays to match the max lengths and then trim after prediction
    event = np.concatenate([df['event'][0], np.full(max_event_len - len(df['event'][0]), -1)]).astype(int)
    time = np.concatenate([df['time'][0], np.full(max_time_len - len(df['time'][0]), 0.)]).astype(float)
    tempo = df['tempo'][0]
    # Reshape the input to match the model input layer
    input_data = np.array([event, time]).reshape((1, max_event_len, 2))
    tempo_pred = model.predict(input_data)
    tempo = tempo_scaler.inverse_transform(tempo.reshape(-1, 1)).flatten()
    tempo_pred = tempo_pred.squeeze()
    tempo_pred = tempo_pred[:len(tempo)]
    tempo_pred = tempo_scaler.inverse_transform(tempo_pred.reshape(-1, 1)).flatten()
    # Convert the tempo to integer arrays
    tempo = np.array([int(x) for x in tempo])
    tempo_pred = np.array([round(x) for x in tempo_pred])
    print("Actual tempo: ", tempo)
    print("Predicted tempo: ", tempo_pred)
    print("Difference: ", tempo - tempo_pred)

    return model, time_scaler, tempo_scaler, max_event_len


def train_time_signature_model(epochs=50):
    """Trains an LSTM model to predict the time signature(s) of a piece based on the events and event times."""
    df = pd.read_csv(f"Data/Tabular/Soprano.csv", sep=';')
    df = df[['event', 'time', 'time_signature_count', 'time_signature_beat']]

    # Normalize the data
    scaler = StandardScaler()
    df['event'] = df['event'].apply(ast.literal_eval).apply(np.array)
    df['time'] = df['time'].apply(ast.literal_eval).apply(np.array)
    df['time_signature_count'] = df['time_signature_beat'].apply(ast.literal_eval).apply(np.array)
    df['time_signature_beat'] = df['time_signature_beat'].apply(ast.literal_eval).apply(np.array)
    df['time_signature_count'] = df['time_signature_count'].apply(lambda x: np.array([int(y) for y in x]))
    df['time_signature_beat'] = df['time_signature_beat'].apply(lambda x: np.array([int(y) for y in x]))
    df = df[df['time'].apply(len) > 0]
    df = df[df['event'].apply(len) > 0]
    df = df[df['time_signature_count'].apply(len) > 0]
    df = df[df['time_signature_beat'].apply(len) > 0]
    df['time'] = df['time'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten())
    inputs = np.array(df[['event', 'time']])
    outputs = np.array(df[['time_signature_count', 'time_signature_beat']])

    # Pad the inputs and outputs to the max length
    max_event_len = max([len(x) for x in inputs[:, 0]])
    max_time_len = max([len(x) for x in inputs[:, 1]])
    max_output_n_len = max([len(x) for x in outputs[:, 0]])
    max_output_d_len = max([len(x) for x in outputs[:, 1]])
    inputs_e = np.array([np.concatenate([x, np.full(max_event_len-len(x), -1)]).astype(int) for x in inputs[:, 0]])
    inputs_t = np.array([np.concatenate([x, np.full(max_time_len-len(x), 0.)]).astype(float) for x in inputs[:, 1]])
    outputs_n = np.array([np.concatenate([x, np.full(max_output_n_len-len(x), 0)]).astype(int) for x in outputs[:, 0]])
    outputs_d = np.array([np.concatenate([x, np.full(max_output_d_len-len(x), 0)]).astype(int) for x in outputs[:, 1]])
    inputs = np.stack((inputs_e, inputs_t), axis=-1)
    outputs = np.stack((outputs_n, outputs_d), axis=-1)

    # Reshape input and output to time signature per GROUP_SIZE notes; must be a factor of max_event_len (25 works well)
    group_size = 25
    inputs = inputs.reshape((-1, group_size, inputs.shape[-1]))
    outputs = outputs.reshape((-1, group_size, outputs.shape[-1]))

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # LSTM
    model = Sequential()
    model.add(layers.LSTM(50, return_sequences=True, input_shape=[None, 2]))
    model.add(layers.LSTM(50, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(2)))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    plot_model(model, to_file=f'Images/time_sig_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', "Time Signature Model Loss (MSE)", 'Loss')
    plot_histories(model, 'mae', 'val_mae', "Time Signature Model MAE", 'MAE')

    # Save the model
    model.save(f"Weights/TimeSignature/model.h5")
    pkl.dump(scaler, open(f"Weights/TimeSignature/time_sig_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights/TimeSignature/seq_len.pkl", 'wb'))

    # Test the model
    event = np.concatenate([df['event'][0], np.full(max_event_len - len(df['event'][0]), -1)]).astype(int)
    time = np.concatenate([df['time'][0], np.full(max_time_len - len(df['time'][0]), 0.)]).astype(float)
    time_sig_counts = df['time_signature_count'][0]
    time_sig_beats = df['time_signature_beat'][0]
    input_data = np.array([event, time]).reshape((1, max_event_len, 2))
    output_data = model.predict(input_data)
    output_data = output_data.squeeze()
    predicted_counts = output_data[:len(time_sig_counts), 0]
    predicted_beats = output_data[:len(time_sig_beats), 1]
    predicted_counts = np.array([round(x) for x in predicted_counts])
    predicted_beats = np.array([round(x) for x in predicted_beats])
    print("Actual counts:", time_sig_counts)
    print("Predicted counts:", predicted_counts)
    print("Difference:", time_sig_counts - predicted_counts)
    print("\nActual beats:", time_sig_beats)
    print("Predicted beats:", predicted_beats)
    print("Difference:", time_sig_beats - predicted_beats)

    return model, scaler, max_event_len


def train_key_model(epochs=10):
    """Trains a Bidirectional LSTM model to predict the key signature(s) of a piece based on the events."""
    df = pd.read_csv(f"Data/Tabular/Soprano.csv", sep=';')
    df = df[['event', 'key_signature']]

    # Normalize the data
    scaler = StandardScaler()
    df['event'] = df['event'].apply(ast.literal_eval).apply(np.array)
    df['key_signature'] = df['key_signature'].apply(ast.literal_eval).apply(np.array)
    df['key_signature'] = df['key_signature'].apply(lambda x: np.array([int(y) for y in x]))
    df = df[df['event'].apply(len) > 0]
    df = df[df['key_signature'].apply(len) > 0]
    df['key_signature'] = df['key_signature'].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten())
    inputs = np.array(df['event'])
    outputs = np.array(df['key_signature'])

    # Pad the inputs and outputs to the max length
    max_event_len = max([len(x) for x in inputs])
    max_output_len = max([len(x) for x in outputs])
    inputs = np.array([np.concatenate([x, np.full(max_event_len-len(x), -1)]).astype(int) for x in inputs])
    outputs = np.array([np.concatenate([x, np.full(max_output_len-len(x), 0)]).astype(int) for x in outputs])

    # Reshape input and output to key signature per GROUP_SIZE notes; must be a factor of max_event_len (25 works well)
    group_size = 25
    inputs = inputs.reshape((-1, group_size, 1))
    outputs = outputs.reshape((-1, group_size, 1))

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Bi-LSTM
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(50, return_sequences=True), input_shape=[None, 1]))
    model.add(layers.Bidirectional(layers.LSTM(50, return_sequences=True)))
    model.add(layers.TimeDistributed(layers.Dense(1, activation='tanh')))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    plot_model(model, to_file=f'Images/key_sig_model.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    plot_histories(model, 'loss', 'val_loss', "Key Signature Model Loss (MSE)", 'Loss')
    plot_histories(model, 'mae', 'val_mae', "Key Signature Model MAE", 'MAE')

    # Save the model
    model.save(f"Weights/KeySignature/model.h5")
    pkl.dump(scaler, open(f"Weights/KeySignature/key_scaler.pkl", 'wb'))
    pkl.dump(max_event_len, open(f"Weights/KeySignature/seq_len.pkl", 'wb'))

    # Test the model
    event = np.concatenate([df['event'][0], np.full(max_event_len - len(df['event'][0]), -1)]).astype(int)
    key_sigs = df['key_signature'][0]
    input_data = np.array([event]).reshape((1, max_event_len, 1))
    output_data = model.predict(input_data)
    output_data = output_data.squeeze()
    predicted_key_sigs = output_data[:len(key_sigs)]
    key_sigs = scaler.inverse_transform(key_sigs.reshape(-1, 1)).flatten()
    key_sigs = np.array([int(x) for x in key_sigs])
    predicted_key_sigs = scaler.inverse_transform(predicted_key_sigs.reshape(-1, 1)).flatten()
    predicted_key_sigs = np.array([round(x) for x in predicted_key_sigs])
    print("Actual key sigs:", key_sigs)
    print("Predicted key sigs:", predicted_key_sigs)
    print("Difference:", key_sigs - predicted_key_sigs)
    print("\nActual keys:", np.array([key_signature_to_number(x) for x in key_sigs]))
    print("Predicted keys:", np.array([key_signature_to_number(x) for x in predicted_key_sigs]))

    return model, scaler, max_event_len

# endregion FeatureModels


if __name__ == "__main__":
    print("Hello, world!")
    # train_tempo_model(epochs=10)
    # train_time_signature_model(epochs=10)
    # train_key_model(epochs=10)
    # train_composition_model("Soprano", epochs=50)
    # train_intro_model(dataset="Tenor", epochs=81)
    # generate_intro(dataset="Soprano", generate_len=30, temperature=0.7)

    # train_markov_composition_model()
    # generate_composition_bpe()
    # train_choral_transformer(epochs=100)

    # key, time_sig, tempo = None, None, None
    # key, time_sig, tempo = "D-:major", "3/4TS", 120
    # for voice_dataset in ["Soprano", "Bass", "Alto", "Tenor"]:
    #     temp = 0.6
    #     g_len = 30
    #     if voice_dataset == "Soprano":
    #         key, time_sig, tempo, _, _ = generate_intro(voice_dataset, generate_len=g_len, temperature=temp,
    #                                                     key=key, time_sig=time_sig, tempo=tempo)
    #     else:
    #         generate_intro(voice_dataset, g_len, temperature=temp, key=key, time_sig=time_sig, tempo=tempo)
    #     # train_duration_model(voice_dataset, epochs=100)
    #     # train_composition_model(voice_dataset, epochs=100)
    #     pass