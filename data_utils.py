import os
import mido
import pandas as pd


def note_number_to_name(note_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = note_number // 12 - 1
    note = note_names[note_number % 12]
    return note + str(octave)


def midi_to_dataframe(midi_file):
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


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), r"Data\MIDI\VoiceParts\Tenor\Isolated\534_001393_tenT.mid")
    df_mid = midi_to_dataframe(data_path)
    pd.set_option('display.max_rows', None)
    print(df_mid)
