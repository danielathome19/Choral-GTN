import os
from music21 import *
from data_utils import validate_and_generate_metatrack


def load_midi(midi_path):
    return converter.parse(midi_path)


# Step 1
def make_notes_diatonic(score, key_signature):
    """
    Transposes notes to be diatonic to the given key signature.
    """
    dt_scale = scale.MajorScale(key_signature) if key_signature.isupper() else scale.MinorScale(key_signature.lower())
    scale_pitches = [p for p in dt_scale.getPitches('A0', 'A9')]
    for part in score.parts:
        for note in part.flat.notes:
            if note.pitch not in scale_pitches:
                # Find the closest diatonic pitch
                closest_pitch = sorted(scale_pitches, key=lambda p: abs(p.midi - note.pitch.midi))[0]
                # Calculate the interval between the current note pitch and the closest diatonic pitch
                interval_to_closest_pitch = interval.Interval(noteStart=note.pitch, noteEnd=closest_pitch)
                # Transpose the note by this interval
                note.transpose(interval_to_closest_pitch, inPlace=True)
    return score


# Step 2
def has_parallel_motion(interval1, interval2):
    """Checks if two intervals are both perfect and in parallel motion (fifths, octaves, unisons)."""
    return interval1.isPerfectConsonance() and interval2.isPerfectConsonance() \
        and interval1.directedName == interval2.directedName


def find_parallel_intervals(parts, interval_type, num_to_compare):
    """
    Find parallel intervals of a given type (like fifths, octaves) between two parts.
    :param      parts: Tuple of two parts to compare (part1, part2)
    :param      interval_type: The type of interval to check for parallels (P5, P8)
    :param      num_to_compare: Number of intervals to check in sequence for parallel motion
    :return:    List of indices where parallel intervals start
    """
    part1, part2 = parts
    parallels = []

    # Flatten the parts to get a single stream of notes
    part1_notes = part1.flat.notes
    part2_notes = part2.flat.notes

    # Get all start times (offsets) for both parts where notes are present
    part1_offsets = [n.offset for n in part1_notes]
    part2_offsets = [n.offset for n in part2_notes]

    # Iterate over all note pairs and compare intervals at offsets present in both parts
    for offset in set(part1_offsets) & set(part2_offsets):
        notes1_at_offset = part1_notes.getElementsByOffset(offset, mustBeginInSpan=False, mustFinishInSpan=False)
        notes2_at_offset = part2_notes.getElementsByOffset(offset, mustBeginInSpan=False, mustFinishInSpan=False)

        # Compare intervals at this offset
        for i in range(len(notes1_at_offset) - num_to_compare + 1):
            current_intervals = [interval.Interval(n1, n2) for n1, n2 in zip(notes1_at_offset[i:i + num_to_compare],
                                                                             notes2_at_offset[i:i + num_to_compare])]

            # Check if all compared intervals are of the specific type and in parallel motion
            if all(ivl.simpleName == interval_type for ivl in current_intervals):
                parallels.append((offset, i))

    return parallels


def correct_notes(part1, part2, index, interval_type):
    """
    A simple correction by moving the second note in the part2 down a step if it's a fifth or octave,
    or up if it's a unison, to break the parallel movement.
    """
    n1 = part1.notes[index]
    n2 = part2.notes[index]

    if interval_type in ['P5', 'P8']:
        # Move n2 down by step to avoid parallel motion, ensure diatonicism.
        new_pitch = n2.pitch.getLowerEnharmonic()
    elif interval_type == 'P1':
        # Move n2 up by step
        new_pitch = n2.pitch.getHigherEnharmonic()
    else:
        return  # If it's neither, no action is taken

    # Ensure the new pitch is diatonic to C major/A minor by simplifying to natural notes
    # TODO: make this work with any key (e.g., same as step 1)
    n2.pitch = new_pitch.getNatural()


def apply_voice_leading(score):
    """
    Applies voice leading rules to the score, correcting parallel fifths, octaves, and unisons.
    """
    soprano, alto, tenor, bass = score.parts
    parts_to_check = [(soprano, alto), (alto, tenor), (tenor, bass)]

    for parts in parts_to_check:
        # Check for parallels in fifths and octaves (can use "P1" for unisons as well)
        for interval_type in ['P5', 'P8', 'P1']:
            parallels = find_parallel_intervals(parts, interval_type, 2)  # Find parallels for each type
            for parallel in parallels:
                offset, index = parallel
                # Correct the parallel notes; however, the simplicity of correction may
                # introduce new voice-leading issues, such as direct fifths or octaves
                correct_notes(parts[0], parts[1], index, interval_type)
    return score


# Step 3: Double correct tones in chords
def double_correct_tones(chord):
    """
    Corrects doubled leading tones, altered tones, and tones in seventh chords within a given chord
    (assuming the chord is within a SATB setting and diatonic to C major/A minor).
    """
    # Assuming the chord is in root position for simplicity; if not, more checks are needed
    root, third, fifth, seventh = None, None, None, None  # Placeholder for chord tones

    # If there's a seventh, it's in the last position in SATB settings for a seventh chord
    if len(chord.pitches) == 4:
        root, third, fifth, seventh = chord.pitches

    # Check for leading tone, which is B in C major/A minor (only check if there's no seventh)
    # TODO: make this work with any key (e.g., same as step 1)
    if not seventh and any(p.name == 'B' for p in chord.pitches):
        # Find all instances of the leading tone
        leading_tones = [p for p in chord.pitches if p.name == 'B']
        if len(leading_tones) > 1:  # If there's a doubled leading tone
            # Move one of them to a different chord tone that's not already in the chord and diatonic
            for p in leading_tones[1:]:  # Skip the first occurrence
                p = move_to_available_pitch(chord, p)

    # Check for altered tones and move them if doubled
    for p in chord.pitches:
        if p.accidental not in (None, 'natural'):  # If the tone is altered
            # We find another note that is not altered and not yet present in the chord to move to
            p = move_to_available_pitch(chord, p)

    # If there is a seventh, ensure it's not doubled
    if seventh and chord.pitches.count(seventh) > 1:
        # Move the doubled seventh to another chord tone not already in the chord and diatonic
        seventh = move_to_available_pitch(chord, seventh)

    return chord


def move_to_available_pitch(chord, pitch_to_move):
    """
    Moves a doubled pitch to the nearest available diatonic pitch that isn't already in the chord.
    """
    diatonic_pitches = ['C', 'D', 'E', 'F', 'G', 'A', 'B']  # TODO: make this work with any key (e.g., same as step 1)
    current_chord_tones = [p.nameWithOctave for p in chord.pitches]
    for diatonic_pitch in diatonic_pitches:
        new_pitch_name_with_octave = diatonic_pitch + str(pitch_to_move.octave)  # Convert octave to string
        if new_pitch_name_with_octave not in current_chord_tones:
            # Return the new pitch with the same octave as the pitch to move
            return note.Note(new_pitch_name_with_octave)

    return pitch_to_move  # If all diatonic pitches are taken, return the original pitch (unlikely in four-part harmony)


# Step 4: Correct melodic intervals
def correct_melodic_intervals(score):
    """
    Corrects melodic intervals within each voice part to adhere to voice-leading rules.
    Checks for augmented seconds, augmented fourths, and large leaps.
    """
    for part in score.parts:
        notes_to_check = []
        for elem in part.flat:
            if isinstance(elem, note.Note):
                notes_to_check.append(elem)

        for i in range(len(notes_to_check) - 1):
            current_note = notes_to_check[i]
            next_note = notes_to_check[i + 1]
            interv = interval.Interval(current_note, next_note)

            # Check for augmented second and tritone and correct them
            if interv.name == 'A2' or interv.name == 'A4' or interv.semitones == 6:  # Augmented second or tritone
                # To correct, change the next note to either a step above or below the current note
                direction = -1 if interv.direction == 'ascending' else 1
                new_next_pitch = pitch.Pitch(current_note.pitch.ps + direction)
                new_next_pitch.octave = next_note.pitch.octave  # Keep the same octave as the next note
                next_note.pitch = new_next_pitch

            # Check for skips larger than an octave or a sixth
            if interv.name in ['m7', 'M7', 'P8'] and (
                    i < len(notes_to_check) - 2):  # Check if there's room for correction
                # The following note should move in stepwise motion in the opposite direction
                following_note = notes_to_check[i + 2]
                stepwise_direction = 1 if interv.direction == 'descending' else -1  # Reverse the direction
                corrected_pitch = pitch.Pitch(next_note.pitch.ps + stepwise_direction)
                corrected_pitch.octave = following_note.pitch.octave
                following_note.pitch = corrected_pitch  # Apply the corrected pitch to the following note

    return score


# Step 5: Correct skips and leaps
def correct_skips_and_leaps(score):
    """
    Corrects skips and leaps within each voice part to adhere to voice-leading rules.
    Any skip larger than a sixth must be followed by stepwise motion in the opposite direction.
    """
    for part in score.parts:
        notes_to_check = []
        for elem in part.flat.notesAndRests:
            if isinstance(elem, note.Note):
                notes_to_check.append(elem)

        for i in range(len(notes_to_check) - 2):  # Iterate with enough lookahead to correct the following note
            current_note = notes_to_check[i]
            next_note = notes_to_check[i + 1]
            following_note = notes_to_check[i + 2]

            # Calculate intervals between current and next, and next and following notes
            skip_interval = interval.Interval(current_note, next_note)
            leap_interval = interval.Interval(next_note, following_note)

            # Correct skips and leaps that are too large
            if skip_interval.isSkip and skip_interval.semitones > 9:  # If skip is greater than a major sixth
                # Determine direction for correction: if skip is ascending, next step should be descending, vice versa
                if skip_interval.direction == "ascending":
                    # Ensuring next interval is a step down
                    if not leap_interval.isStep or leap_interval.direction == "ascending":
                        corrected_pitch = pitch.Pitch(next_note.pitch.ps - 2)  # Step down in pitch space (MIDI number)
                        following_note.pitch = corrected_pitch
                else:
                    # Ensuring next interval is a step up
                    if not leap_interval.isStep or leap_interval.direction == "descending":
                        corrected_pitch = pitch.Pitch(next_note.pitch.ps + 2)  # Step up in pitch space
                        following_note.pitch = corrected_pitch

    return score


# Step 6: Handle dissonances correctly
def handle_dissonances(score):
    """
    Resolves dissonances by ensuring that:
    - Sevenths resolve down by step.
    - A perfect fourth against the bass only occurs as part of the third inversion of a seventh chord.
    """
    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            for i, chord in enumerate(measure.getElementsByClass('Chord')):
                # Process sevenths resolving down by step
                if i > 0:  # If not the first chord, check the previous chord for sevenths to resolve
                    previous_chord = measure.getElementsByClass('Chord')[i-1]
                    if previous_chord.seventh:  # Check if the previous chord has a seventh
                        seventh_note = previous_chord.seventh
                        seventh_index = previous_chord.pitches.index(seventh_note)
                        # Resolve the seventh down by step
                        resolved_pitch = pitch.Pitch(seventh_note.midi - 1)
                        chord.pitches[seventh_index] = resolved_pitch

                # Process perfect fourths
                if chord.isTriad() and interval.Interval(chord.bass(), chord.pitches[1]).simpleName == 'P4':
                    # Check if the fourth is allowed, if not, alter it
                    if chord.inversion() != 3:  # Ensure the fourth is allowed only in third inversion chords
                        tenor_pitch = chord.pitches[1]  # The second pitch in SATB ordering
                        # Alter the tenor pitch to make a consonant interval with the bass
                        new_tenor_interval = interval.Interval(chord.bass(), tenor_pitch).transpose('P5')
                        chord.pitches[1] = new_tenor_interval.noteStart

    return score


def get_new_bpm():
    t_key, t_time_sig, t_tempo, entrance = validate_and_generate_metatrack('Soprano')
    return t_tempo


def change_bpm_and_key(midi_file_path, new_bpm, output_file_path):
    mf = midi.MidiFile()
    mf.open(midi_file_path)
    mf.read()
    mf.close()
    midi_stream = midi.translate.midiFileToStream(mf)

    # Remove existing tempo changes
    for el in midi_stream.flat.getElementsByClass(tempo.MetronomeMark):
        midi_stream.remove(el, recurse=True)

    new_metronome_mark = tempo.MetronomeMark(number=new_bpm)
    midi_stream.insert(0, new_metronome_mark)

    # Change key signature
    for el in midi_stream.recurse():
        if isinstance(el, key.KeySignature):
            if el.mode == "minor":
                el.tonic = key.pitch.Pitch('A')
            else:
                el.tonic = key.pitch.Pitch('C')

    midi_stream.write('midi', fp=output_file_path)


def save_midi(score, adjusted_midi_path):
    score.write('midi', fp=adjusted_midi_path)


def process_midi(midi_path, adjusted_midi_path='Data/Postprocessed', verbose=True):
    score = load_midi(midi_path)
    print("Loaded MIDI file; beginning post-processing...")
    score = make_notes_diatonic(score, 'C')
    print("Completed step 1\n" if verbose else "", end="")
    score = apply_voice_leading(score)
    print("Completed step 2\n" if verbose else "", end="")
    score = double_correct_tones(score)
    print("Completed step 3\n" if verbose else "", end="")
    score = correct_melodic_intervals(score)
    print("Completed step 4\n" if verbose else "", end="")
    score = correct_skips_and_leaps(score)
    print("Completed step 5\n" if verbose else "", end="")
    score = handle_dissonances(score)
    print("Completed step 6\n" if verbose else "", end="")
    print("Rerunning steps 1 and 2 to adjust any new voice leading issues...\n" if verbose else "", end="")
    score = apply_voice_leading(score)
    score = make_notes_diatonic(score, 'C')
    print("Completed rerun of steps 1 and 2\n" if verbose else "", end="")
    if not os.path.exists(adjusted_midi_path):
        os.makedirs(adjusted_midi_path)
    output_path = os.path.join(adjusted_midi_path, os.path.basename(midi_path))
    save_midi(score, output_path)
    print(f"Saved post-processed MIDI file to {output_path}")
    return output_path


if __name__ == "__main__":
    print("Hello, world!")
    bpm_option = input("Would you like to change the BPM of the MIDI file? "
                       "(\033[4mn\033[0mo/\033[4mr\033[0mandom/enter number [e.g., 160]): ")
    if bpm_option.lower() in "no":
        bpm = None
    elif bpm_option.lower() in "random":
        bpm = int(get_new_bpm().split('BPM')[0])
        print(f"Randomly selected BPM: {bpm}")
    else:
        bpm = int(bpm_option)
    path = input("Enter the path to the MIDI file: ")
    output = process_midi(path)
    if bpm is not None:
        change_bpm_and_key(output, bpm, output)
        print("Finished changing BPM.")
    # TODO: add option to change key (based on current mode)
    pass
