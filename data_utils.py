import os
import mido
import pandas as pd


def note_number_to_name(note_number):
    """Converts a MIDI note number to a note name with pitch class."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = note_number // 12 - 1
    note = note_names[note_number % 12]
    return note + str(octave)


def midi_to_dataframe(midi_file):
    """Converts a MIDI file to a pandas dataframe.
    The dataframe has the following columns:
    - event: The name of the note or rest
    - velocity: The velocity of the note
    - time: The time in seconds of the event
    - tempo: The tempo in beats per minute at the time of the event
    - time_signature: The time signature at the time of the event
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
                        events.append('rest')
                        velocities.append(0)
                        times.append(last_event_time)
                        tempi.append(mido.tempo2bpm(current_tempo))
                        time_signatures.append(current_time_signature)
                        key_signatures.append(current_key_signature)
                    events.append(note_number_to_name(msg.note))
                    velocities.append(msg.velocity)
                    times.append(current_time)
                    tempi.append(mido.tempo2bpm(current_tempo))
                    time_signatures.append(current_time_signature)
                    key_signatures.append(current_key_signature)
                    last_event_time = current_time
            elif msg.type == 'set_tempo':
                current_tempo = msg.tempo
            elif msg.type == 'time_signature':
                current_time_signature = f"{msg.numerator}/{msg.denominator}"
            elif msg.type == 'key_signature':
                current_key_signature = msg.key
        current_time = 0

    df = pd.DataFrame({'event': events, 'velocity': velocities, 'time': times, 'tempo': tempi,
                       'time_signature': time_signatures, 'key_signature': key_signatures})
    return df


def transpose_df_to_row(dataframe):
    """Transpose a dataframe to a single row where each column is a 1D array from the original dataframe."""
    df = pd.DataFrame()
    for column in dataframe.columns:
        df[column] = [dataframe[column].to_numpy()]
    return df


def build_dataset(data_dir):
    """Builds a dataset from a directory of MIDI files."""
    df = pd.DataFrame()
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                df = pd.concat([df, transpose_df_to_row(midi_to_dataframe(os.path.join(root, file)))])
    return df


def save_dataset_csv(dataframe, output_file):
    dataframe.to_csv(output_file, index=False)


if __name__ == "__main__":
    # data_path = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Tenor\Isolated\534_001393_tenT.mid")
    # df_mid = midi_to_dataframe(data_path)
    # pd.set_option('display.max_rows', None)
    # print(df_mid)
    # print(transpose_df_to_row(df_mid))

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
    save_dataset_csv(df_soprano, os.path.join(os.getcwd(), r"Data\Tabular\Soprano.csv"))
    save_dataset_csv(df_alto, os.path.join(os.getcwd(), r"Data\Tabular\Alto.csv"))
    save_dataset_csv(df_tenor, os.path.join(os.getcwd(), r"Data\Tabular\Tenor.csv"))
    save_dataset_csv(df_bass, os.path.join(os.getcwd(), r"Data\Tabular\Bass.csv"))

    print("Complete!")