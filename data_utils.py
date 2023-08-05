import glob
import os
import mido
import music21
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from keras import layers
from fractions import Fraction


# region Dataframes
def note_number_to_name(note_number):
    """Converts a MIDI note number to a note name with pitch class."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = note_number // 12 - 1
    note = note_names[note_number % 12]
    return note + str(octave)


def key_signature_to_number(key_signature):
    mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
               'A#', 'B#', 'C#', 'D#', 'E#', 'F#', 'G#',
               'Ab', 'Bb', 'Cb', 'Db', 'Eb', 'Fb', 'Gb',
               'Am', 'Bm', 'Cm', 'Dm', 'Em', 'Fm', 'Gm',
               'A#m', 'B#m', 'C#m', 'D#m', 'E#m', 'F#m', 'G#m',
               'Abm', 'Bbm', 'Cbm', 'Dbm', 'Ebm', 'Fbm', 'Gbm']
    if str(key_signature).isnumeric():
        return mapping[int(key_signature)]
    return mapping.index(key_signature)


def midi_to_dataframe(midi_file):
    """Converts a MIDI file to a pandas dataframe.
    The dataframe has the following columns:
    - event: The name of the note or rest
    - velocity: The velocity of the note
    - time: The time in seconds of the event
    - tempo: The tempo in beats per minute at the time of the event
    - time_signature_count: The time signature numerator at the time of the event
    - time_signature_beat: The time signature denominator at the time of the event
    - key_signature: The key signature at the time of the event
    """
    mid = mido.MidiFile(midi_file)

    events = []
    velocities = []
    times = []
    tempi = []
    time_signatures = []
    key_signatures = []

    current_tempo = 500000  # MIDI default tempo (microseconds per beat)
    current_time_signature = '4/4'  # Default time signature
    current_key_signature = 'C'  # Default key signature
    current_time = 0  # Current time in seconds
    last_event_time = 0  # Time of the last event

    for _, track in enumerate(mid.tracks):
        for i, msg in enumerate(track):
            time_delta = mido.tick2second(msg.time, mid.ticks_per_beat, current_tempo)
            current_time += time_delta

            if msg.type == 'note_on':
                if msg.velocity > 0:
                    if current_time > last_event_time:
                        # There is a gap between the last event and this one, insert a rest
                        # events.append('rest')
                        events.append(-1)
                        velocities.append(0)
                        times.append(last_event_time)
                        tempi.append(mido.tempo2bpm(current_tempo))
                        time_signatures.append(current_time_signature)
                        # key_signatures.append(current_key_signature)
                        key_signatures.append(key_signature_to_number(current_key_signature))
                    # events.append(note_number_to_name(msg.note))
                    events.append(msg.note)
                    velocities.append(msg.velocity)
                    times.append(current_time)
                    tempi.append(mido.tempo2bpm(current_tempo))
                    time_signatures.append(current_time_signature)
                    # key_signatures.append(current_key_signature)
                    key_signatures.append(key_signature_to_number(current_key_signature))
                    last_event_time = current_time
            elif msg.type == 'set_tempo':
                current_tempo = msg.tempo  # May need to record in a meta_events list also
            elif msg.type == 'time_signature':
                current_time_signature = f"{msg.numerator}/{msg.denominator}"
            elif msg.type == 'key_signature':
                current_key_signature = msg.key
        current_time = 0

    # Split the time signature into two arrays, one for the numerator and one for the denominator
    time_signature_counts = []
    time_signature_beats = []

    for time_signature in time_signatures:
        time_signature_counts.append(time_signature.split('/')[0])
        time_signature_beats.append(time_signature.split('/')[1])

    df = pd.DataFrame({'event': events, 'velocity': velocities, 'time': times, 'tempo': tempi,
                       'time_signature_count': time_signature_counts, 'time_signature_beat': time_signature_beats,
                       'key_signature': key_signatures})
    return df


def transpose_df_to_row(dataframe):
    """Transpose a dataframe to a single row where each column is a 1D array from the original dataframe."""
    np.set_printoptions(threshold=np.inf)
    df = pd.DataFrame()
    for column in dataframe.columns:
        # Turn each column into a 1D array, then turn the array into a string in the form "[1, 2, 3, ...]"
        df[column] = [np.array2string(dataframe[column].to_numpy(), separator=',')]
    return df


def build_dataset(data_dir):
    """Builds a dataset from a directory of MIDI files."""
    df = pd.DataFrame()
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                df = pd.concat([df, transpose_df_to_row(midi_to_dataframe(os.path.join(root, file)))])
    return df


def save_dataset(dataframe, output_file):
    dataframe.to_csv(output_file, index=False, sep=';')
# endregion Dataframes


# region VoiceTransformer
def parse_midi_files(file_list, parser, seq_len, parsed_data_path=None, verbose=False, enable_chords=False, limit=None):
    notes = []
    durations = []

    if limit is not None:
        file_list = file_list[:limit]
    for i, file in enumerate(file_list):
        if verbose:
            print(i + 1, "Parsing %s" % file)
        score = parser.parse(file).chordify()
        notes.append("START")
        durations.append("0.0")
        for element in score.flat:
            note_name = None
            duration_name = None
            # TODO: add instance for clef?
            if isinstance(element, music21.key.Key):
                note_name = str(element.tonic.name) + ":" + str(element.mode)
                duration_name = "0.0"
            elif isinstance(element, music21.meter.TimeSignature):
                note_name = str(element.ratioString) + "TS"
                duration_name = "0.0"
            elif isinstance(element, music21.chord.Chord):
                note_name = '.'.join(n.nameWithOctave for n in element.pitches) if enable_chords \
                            else element.pitches[-1].nameWithOctave
                duration_name = str(element.duration.quarterLength)
            elif isinstance(element, music21.note.Rest):
                note_name = str(element.name)
                duration_name = str(element.duration.quarterLength)
            elif isinstance(element, music21.note.Note):
                note_name = str(element.nameWithOctave)
                duration_name = str(element.duration.quarterLength)
            if note_name and duration_name:
                notes.append(note_name)
                durations.append(duration_name)
        if verbose:
            print(f"{len(notes)} notes parsed")

    notes_list = []
    duration_list = []
    if verbose:
        print(f"Building sequences of length {seq_len}")
    for i in range(len(notes) - seq_len):
        notes_list.append(" ".join(notes[i: (i + seq_len)]))
        duration_list.append(" ".join(durations[i: (i + seq_len)]))
    if parsed_data_path:
        with open((parsed_data_path + "notes.pkl"), "wb") as f:
            pkl.dump(notes_list, f)
        with open((parsed_data_path + "durations.pkl"), "wb") as f:
            pkl.dump(duration_list, f)

    return notes_list, duration_list


def create_transformer_dataset(elements, batch_size=256):
    ds = (tf.data.Dataset.from_tensor_slices(elements).batch(batch_size, drop_remainder=True).shuffle(1000))
    vectorize_layer = layers.TextVectorization(standardize=None, output_mode="int")
    vectorize_layer.adapt(ds)
    vocab = vectorize_layer.get_vocabulary()
    return ds, vectorize_layer, vocab


def load_parsed_files(parsed_data_path, from_slices=False):
    if from_slices:
        notes = load_pickle_from_slices(parsed_data_path + "notes")
        durations = load_pickle_from_slices(parsed_data_path + "durations")
        return notes, durations
    with open((parsed_data_path + "notes.pkl"), "rb") as f:
        notes = pkl.load(f)
    with open((parsed_data_path + "durations.pkl"), "rb") as f:
        durations = pkl.load(f)
    return notes, durations


def get_midi_note(sample_note, sample_duration):
    new_note = None
    if "TS" in sample_note:
        new_note = music21.meter.TimeSignature(sample_note.split("TS")[0])
    elif "major" in sample_note or "minor" in sample_note:
        tonic, mode = sample_note.split(":")
        new_note = music21.key.Key(tonic, mode)
    elif sample_note == "rest":
        new_note = music21.note.Rest()
        new_note.duration = music21.duration.Duration(float(Fraction(sample_duration)))
        new_note.storedInstrument = music21.instrument.Vocalist()
    elif "." in sample_note:
        notes_in_chord = sample_note.split(".")
        chord_notes = []
        for current_note in notes_in_chord:
            n = music21.note.Note(current_note)
            n.duration = music21.duration.Duration(float(Fraction(sample_duration)))
            n.storedInstrument = music21.instrument.Vocalist()
            chord_notes.append(n)
        new_note = music21.chord.Chord(chord_notes)
    elif sample_note == "rest":
        new_note = music21.note.Rest()
        new_note.duration = music21.duration.Duration(float(Fraction(sample_duration)))
        new_note.storedInstrument = music21.instrument.Vocalist()
    elif sample_note != "START":
        new_note = music21.note.Note(sample_note)
        new_note.duration = music21.duration.Duration(float(Fraction(sample_duration)))
        new_note.storedInstrument = music21.instrument.Vocalist()
    return new_note


def compile_midi_from_voices():
    soprano_path = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Soprano\Isolated")
    alto_path = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Alto\Isolated")
    tenor_path = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Tenor\Isolated")
    bass_path = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Bass\Isolated")
    soprano_files = sorted([f for f in os.listdir(soprano_path) if f.lower().endswith('.mid') and f != 'desktop.ini'])
    alto_files = sorted([f for f in os.listdir(alto_path) if f.lower().endswith('.mid') and f != 'desktop.ini'])
    tenor_files = sorted([f for f in os.listdir(tenor_path) if f.lower().endswith('.mid') and f != 'desktop.ini'])
    bass_files = sorted([f for f in os.listdir(bass_path) if f.lower().endswith('.mid') and f != 'desktop.ini'])
    for soprano_file, alto_file, tenor_file, bass_file in zip(soprano_files, alto_files, tenor_files, bass_files):
        new_midi = mido.MidiFile()
        soprano_midi = mido.MidiFile(os.path.join(soprano_path, soprano_file))
        new_midi.tracks.append(soprano_midi.tracks[0])
        new_midi.ticks_per_beat = soprano_midi.ticks_per_beat
        for voice_path, voice_file in zip([soprano_path, alto_path, tenor_path, bass_path],
                                          [soprano_file, alto_file, tenor_file, bass_file]):
            voice_midi = mido.MidiFile(os.path.join(voice_path, voice_file))
            for i, track in enumerate(voice_midi.tracks):
                if i != 0:
                    new_midi.tracks.append(track)
                elif i == 0 and voice_file != soprano_file:
                    for msg in track:
                        new_midi.tracks[0].append(msg)
        filename = soprano_file.split(".")[0] + "_all.mid"
        new_midi.save(os.path.join(os.getcwd(), "Data\\MIDI\\VoiceParts\\Combined", filename))
        print("Saved file: " + filename)
    pass


def slice_pickle(path, slices=4):
    """Slices a pickle file into smaller pieces for easier uploading to GitHub."""
    with open(path, "rb") as f:
        data = pkl.load(f)
        print("Found data of length:", len(data))
    slice_size = len(data) // slices
    for i in range(slices):
        start_index = i * slice_size
        end_index = (i + 1) * slice_size if i != slices - 1 else len(data)
        slice_data = data[start_index:end_index]
        base_name = os.path.basename(path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(os.path.dirname(path), f"{name}_{i + 1}{ext}")
        with open(output_path, 'wb') as f:
            pkl.dump(slice_data, f)
        print(f"Saved slice {i + 1} to {output_path}")


def load_pickle_from_slices(filename):
    """Loads a pickle file that has been sliced into smaller pieces for easier uploading to GitHub."""
    dir_name = os.path.dirname(filename)
    base_name = os.path.basename(filename)
    # name, ext = os.path.splitext(base_name)
    slice_files = sorted(glob.glob(os.path.join(dir_name, f"{base_name}_*.pkl")))  # {name}_*{ext}
    if not slice_files:
        raise ValueError(f"No sliced pickle files found for {filename}")
    combined_data = []
    for slice_file in slice_files:
        with open(slice_file, 'rb') as f:
            slice_data = pkl.load(f)
            combined_data.extend(slice_data)
    print("Loaded data of length:", len(combined_data))
    return combined_data

# endregion VoiceTransformer


if __name__ == "__main__":
    # data_path = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Tenor\Isolated\534_001393_tenT.mid")
    # df_mid = midi_to_dataframe(data_path)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(df_mid)
    # print(transpose_df_to_row(df_mid))
    # compile_midi_from_voices()
    # slice_pickle("Data\\Glob\\Combined\\Combined_notes.pkl")
    # slice_pickle("Data\\Glob\\Combined\\Combined_durations.pkl")
    # load_pickle_from_slices("Data\\Glob\\Combined\\Combined_notes")
    # load_pickle_from_slices("Data\\Glob\\Combined\\Combined_durations")
    # quit()

    SOPRANO_PATH = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Soprano\Isolated")
    ALTO_PATH = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Alto\Isolated")
    TENOR_PATH = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Tenor\Isolated")
    BASS_PATH = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Bass\Isolated")

    print("Building datasets...")
    df_soprano = build_dataset(SOPRANO_PATH)
    df_alto = build_dataset(ALTO_PATH)
    df_tenor = build_dataset(TENOR_PATH)
    df_bass = build_dataset(BASS_PATH)

    print("Saving datasets...")
    save_dataset(df_soprano, os.path.join(os.getcwd(), r"Data\Tabular\Soprano.csv"))
    save_dataset(df_alto, os.path.join(os.getcwd(), r"Data\Tabular\Alto.csv"))
    save_dataset(df_tenor, os.path.join(os.getcwd(), r"Data\Tabular\Tenor.csv"))
    save_dataset(df_bass, os.path.join(os.getcwd(), r"Data\Tabular\Bass.csv"))

    print("Complete!")
