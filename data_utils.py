import os
import glob
import mido
import music21
import pretty_midi
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


def create_all_datasets():
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

# endregion Dataframes


# region VoiceTransformer
def parse_choral_midi_files(file_list, parser, seq_len, parsed_data_path=None, verbose=False, limit=None, mm_limit=0):
    all_voices_data = {'Soprano': [], 'Alto': [], 'Tenor': [], 'Bass': []}

    if limit is not None:
        file_list = file_list[:limit]

    for i, file in enumerate(file_list):
        if verbose:
            print(i + 1, "Parsing %s" % file)
        score = parser.parse(file)
        for part, voice in zip(score.parts, all_voices_data.keys()):
            notes = ["START"]
            durations = ["0.0"]
            if mm_limit != 0:
                part = part.measures(0, mm_limit)
            for element in part.flat:
                note_name = None
                duration_name = None
                if isinstance(element, music21.tempo.MetronomeMark):
                    note_name = str(element.number) + "BPM"
                    duration_name = "0.0"
                elif isinstance(element, music21.key.Key):
                    note_name = str(element.tonic.name) + ":" + str(element.mode)
                    duration_name = "0.0"
                elif isinstance(element, music21.meter.TimeSignature):
                    note_name = str(element.ratioString) + "TS"
                    duration_name = "0.0"
                elif isinstance(element, music21.note.Rest):
                    note_name = voice + ":" + str(element.name)
                    duration_name = str(element.duration.quarterLength)
                elif isinstance(element, music21.note.Note):
                    note_name = voice + ":" + str(element.nameWithOctave)
                    duration_name = str(element.duration.quarterLength)
                if note_name and duration_name:
                    notes.append(note_name)
                    durations.append(duration_name)
            notes.append("END")
            durations.append("0.0")
            for j in range(len(notes) - seq_len):
                all_voices_data[voice].append({
                    'notes': " ".join(notes[j: (j + seq_len)]),
                    'durations': " ".join(durations[j: (j + seq_len)])
                })
    if parsed_data_path:
        for voice, data in all_voices_data.items():
            with open((parsed_data_path + f"{voice}_choral_notes.pkl"), "wb") as f:
                pkl.dump([entry['notes'] for entry in data], f)
            with open((parsed_data_path + f"{voice}_choral_durations.pkl"), "wb") as f:
                pkl.dump([entry['durations'] for entry in data], f)
    return all_voices_data


def get_choral_midi_note(sample_token, sample_duration):
    new_note = None
    voice_type, sample_note = sample_token.split(":")[0], ":".join(sample_token.split(":")[1:])
    if "BPM" in sample_token:
        new_note = music21.tempo.MetronomeMark(number=int(sample_token.split("BPM")[0]))
    elif "TS" in sample_token:
        new_note = music21.meter.TimeSignature(sample_token.split("TS")[0])
    elif "major" in sample_note or "minor" in sample_note:
        tonic, mode = sample_token.split(":")
        new_note = music21.key.Key(tonic, mode)
    elif sample_note == "rest":
        new_note = music21.note.Rest()
        new_note.duration = music21.duration.Duration(float(Fraction(sample_duration)))
        new_note.storedInstrument = get_voice_instrument(voice_type)
    elif sample_note != "START" and sample_note != "END":
        new_note = music21.note.Note(sample_note)
        new_note.duration = music21.duration.Duration(float(Fraction(sample_duration)))
        new_note.storedInstrument = get_voice_instrument(voice_type)
    return new_note


def get_voice_instrument(voice_type):
    if voice_type == "Soprano":
        return music21.instrument.Soprano()
    elif voice_type == "Alto":
        return music21.instrument.Alto()
    elif voice_type == "Tenor":
        return music21.instrument.Tenor()
    elif voice_type == "Bass":
        return music21.instrument.Bass()
    else:
        return music21.instrument.Vocalist()


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
            # if isinstance(element, music21.clef.Clef):
            #     note_name = f"{element.sign}:{element.line}:{element.octaveChange}CLEF"
            #     duration_name = "0.0"
            if isinstance(element, music21.tempo.MetronomeMark):
                note_name = str(element.number) + "BPM"
                duration_name = "0.0"
            elif isinstance(element, music21.key.Key):
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


def get_midi_note(sample_note, sample_duration):
    new_note = None
    # if "CLEF" in sample_note:
    #     sign, line, octave_change = sample_note.split("CLEF")[0].split(":")
    #     new_note = music21.clef.Clef(sign=sign, line=int(line), octaveChange=int(octave_change))
    if "BPM" in sample_note:
        new_note = music21.tempo.MetronomeMark(number=round(float(sample_note.split("BPM")[0])))
    elif "TS" in sample_note:
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
        new_midi.save(os.path.join(os.getcwd(), "Data/MIDI/VoiceParts/Combined", filename))
        print("Saved file: " + filename)
    pass


def augment_midi_files(path):
    """Augments MIDI file dataset using the following methods:
    - Transpose by # semitones (with adjusted key signature to match)
    - Speed up/slow down tempo (adjust bpm)
    """
    MAJOR_COF = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
    MINOR_COF = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'Bb', 'F', 'C', 'G', 'D']

    def adjust_key_signature(midi_file, semitones):
        for key_change in midi_file.key_signature_changes:
            current_key = key_change.key_number
            is_major = current_key < 12
            current_key_name = MAJOR_COF[current_key] if is_major else MINOR_COF[current_key - 12]
            if is_major:
                new_key_index = (MAJOR_COF.index(current_key_name) + semitones) % 12
                new_key_name = MAJOR_COF[new_key_index]
                new_key_number = MAJOR_COF.index(new_key_name)
            else:
                new_key_index = (MINOR_COF.index(current_key_name) + semitones) % 12
                new_key_name = MINOR_COF[new_key_index]
                new_key_number = MINOR_COF.index(new_key_name) + 12
            key_change.key_number = new_key_number
        return midi_file

    def adjust_pitch(midi_file, semitones):
        for instrument in midi_file.instruments:
            for note in instrument.notes:
                note.pitch += semitones
                # Ensure the pitch remains within MIDI bounds (0-127)
                note.pitch = min(max(note.pitch, 0), 127)
        return midi_file

    def adjust_tempo(midi_file_path, factor):
        midi_file = mido.MidiFile(midi_file_path)
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    msg.tempo = int(msg.tempo * factor)
        midi_file.save(midi_file_path)
    
    files = sorted([f for f in os.listdir(path) if f.lower().endswith('.mid')
                    or f.lower().endswith('.midi') and f != 'desktop.ini'])
    tempo_adjustments = [1.3, 0.6, 2.0, 0.9]

    for file in files:
        print("Augmenting file: " + file)
        midi = pretty_midi.PrettyMIDI(os.path.join(path, file))
        for i in range(1, 5):
            new_midi = midi
            if i == 1:
                # Shift down 6 half-steps (tritone), slow down all durations by 30%
                new_midi = adjust_pitch(new_midi, -6)
                new_midi = adjust_key_signature(new_midi, -6)
            elif i == 2:
                # Shift up 4 half-steps (major 3rd), speed up all BPMs by 40%
                new_midi = adjust_pitch(new_midi, 4)
                new_midi = adjust_key_signature(new_midi, 4)
            elif i == 3:
                # Shift up perfect 4th, slow down all BPMs by 50%
                new_midi = adjust_pitch(new_midi, 5)
                new_midi = adjust_key_signature(new_midi, 5)
            elif i == 4:
                # Shift down minor 6th, speed up all BPMs by 20%
                new_midi = adjust_pitch(new_midi, -8)
                new_midi = adjust_key_signature(new_midi, -8)
            # elif i == 5:
            # Shift up minor 2nd, speed up all BPMs by 60% (0.4)
            if not os.path.exists(os.path.join(path, f"Augment_{i}")):
                os.makedirs(os.path.join(path, f"Augment_{i}"))
            output_path = os.path.join(path, f"Augment_{i}", f"{os.path.splitext(file)[0]}_aug{i}.mid")
            new_midi.write(output_path)
            tempo_factor = tempo_adjustments[i - 1]
            adjust_tempo(output_path, tempo_factor)
    pass


def glob_midis(path, output_path="Data/Glob/Combined/Combined_", suffix="", choral=False, measure_limit=0):
    SEQ_LEN = 50
    POLYPHONIC = True
    file_list = glob.glob(path + "/*.mid")
    parser = music21.converter
    print(f"Parsing {len(file_list)} midi files...")
    if not choral:
        _, _ = parse_midi_files(file_list, parser, SEQ_LEN + 1, output_path + suffix,
                                verbose=True, enable_chords=POLYPHONIC, limit=None)
    else:
        _ = parse_choral_midi_files(file_list, parser, SEQ_LEN + 1, output_path + suffix,
                                    verbose=True, limit=None, mm_limit=measure_limit)
    print("Complete!")


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


def load_pickle_from_slices(filename, include_augmented=False):
    """Loads a pickle file that has been sliced into smaller pieces for easier uploading to GitHub."""
    dir_name = os.path.dirname(filename)
    base_name = os.path.basename(filename)
    # name, ext = os.path.splitext(base_name)
    slice_files = sorted(glob.glob(os.path.join(dir_name, f"{base_name}_*.pkl")))  # {name}_*{ext}
    if include_augmented:
        base, dset = base_name.split("_")[:2]
        slice_files.extend(sorted(glob.glob(os.path.join(dir_name, f"{base}_aug*{dset}*.pkl"))))
        slice_files = sorted(slice_files)
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
    print("Hello, world!")
    # data_path = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Tenor\Isolated\534_001393_tenT.mid")
    # df_mid = midi_to_dataframe(data_path)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(df_mid)
    # print(transpose_df_to_row(df_mid))
    # create_all_datasets()
    # compile_midi_from_voices()
    # slice_pickle("Data/Glob/Combined/Combined_notes.pkl")
    # slice_pickle("Data/Glob/Combined/Combined_durations.pkl")
    # load_pickle_from_slices("Data/Glob/Combined/Combined_notes")
    # load_pickle_from_slices("Data/Glob/Combined/Combined_durations")
    # augment_midi_files("Data/MIDI/VoiceParts/Combined")
    # glob_midis("Data/MIDI/VoiceParts/Combined", "Data/Glob/Combined/Combined_")
    # slice_pickle("Data/Glob/Combined/Combined_notes.pkl", slices=5)
    # slice_pickle("Data/Glob/Combined/Combined_durations.pkl", slices=5)
    """
    for i in range(1, 5):
        glob_midis("Data/MIDI/VoiceParts/Combined/Augment_1", "Data/Glob/Combined/Combined_aug1_")
        glob_midis("Data/MIDI/VoiceParts/Combined/Augment_2", "Data/Glob/Combined/Combined_aug2_")
        glob_midis("Data/MIDI/VoiceParts/Combined/Augment_3", "Data/Glob/Combined/Combined_aug3_")
        glob_midis("Data/MIDI/VoiceParts/Combined/Augment_4", "Data/Glob/Combined/Combined_aug4_")
        slice_pickle(f"Data/Glob/Combined/Combined_aug{i}_notes.pkl")
        slice_pickle(f"Data/Glob/Combined/Combined_aug{i}_durations.pkl")
    """
    # load_pickle_from_slices("Data/Glob/Combined/Combined_notes", True)
    # load_pickle_from_slices("Data/Glob/Combined/Combined_durations", True)
    # glob_midis("Data/MIDI/VoiceParts/Combined", "Data/Glob/Combined_choral/Combined_", choral=True)
    # for voice in ["Soprano", "Alto", "Tenor", "Bass"]:
    #     slice_pickle(f"Data/Glob/Combined_choral/Combined_{voice}_choral_notes.pkl", slices=5)
    #     slice_pickle(f"Data/Glob/Combined_choral/Combined_{voice}_choral_durations.pkl", slices=5)

    glob_midis("Data/MIDI/VoiceParts/Combined", "Data/Glob/Combined_mm1-8/Combined_", choral=True, measure_limit=8)
    for i in range(1, 5):
        glob_midis("Data/MIDI/VoiceParts/Combined/Augment_1", "Data/Glob/Combined_mm1-8/Combined_aug1_", "", True, 8)
        glob_midis("Data/MIDI/VoiceParts/Combined/Augment_2", "Data/Glob/Combined_mm1-8/Combined_aug2_", "", True, 8)
        glob_midis("Data/MIDI/VoiceParts/Combined/Augment_3", "Data/Glob/Combined_mm1-8/Combined_aug3_", "", True, 8)
        glob_midis("Data/MIDI/VoiceParts/Combined/Augment_4", "Data/Glob/Combined_mm1-8/Combined_aug4_", "", True, 8)
    pass
