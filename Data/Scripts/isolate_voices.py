# For ever folder in the VoiceParts directory, iterate through each MIDI file in the folder.
# In each folder, make a subfolder called "Isolated".
# For each MIDI file, if there is only one voice part, copy the file to the "Isolated" subfolder.
# Otherwise, if there are multiple voices in the MIDI, isolate the intended voice and copy it to the "Isolated" folder.

import os
import mido
import shutil

count = 0
current_dir = os.path.join(os.getcwd(), "Data", "MIDI", "VoiceParts")
for folder in os.listdir(current_dir):
    broken_files = []
    folder_path = os.path.join(current_dir, folder)
    for file in os.listdir(folder_path):
        try: 
            if file.split('.')[1].lower().strip() != "mid" and file.split('.')[1].lower().strip() != "midi": continue
        except IndexError as e: continue
        file_path = os.path.join(folder_path, file)
        isolated_folder_path = os.path.join(folder_path, "Isolated")
        if not os.path.exists(isolated_folder_path):
            os.mkdir(isolated_folder_path)
        
        try:
            midi_file = mido.MidiFile(file_path)
        except Exception as e:
            print(f"\t\tError processing {file}: {e}")
            broken_files.append(file)
            continue

        tracks = [track for track in midi_file.tracks if not all(msg.type == 'sysex' for msg in track)]
        # print(f"Processing {file} with {len(tracks)} tracks")
        if midi_file.type != 1:
            print(f"\t\tFile {file} is not a type 1 MIDI file; possible error")
        if len(tracks) <= 2:
            # If there is only one track (and the meta track), copy the file to the "Isolated" folder as is
            shutil.copy(file_path, isolated_folder_path)
        else:
            voice_index = None
            # Iterate through each track name and find the one closest to the voice part
            for i, track in enumerate(tracks):
                if track.name.lower().strip() == folder.lower() or \
                   track.name.lower()[:3] == folder.lower()[:3] or \
                   track.name.lower()[:len(folder)] == folder.lower() or \
                   len(track.name.strip()) == 1 and track.name.lower() == folder.lower()[0] or \
                   "treble" in track.name.lower() and folder.lower() == "soprano" or \
                   "cantus" in track.name.lower() and folder.lower() == "soprano" or \
                   "countertenor" in track.name.lower() and folder.lower() == "alto" or \
                   "altus" in track.name.lower() and folder.lower() == "alto" or \
                   "bassus" in track.name.lower() and folder.lower() == "bass":
                    voice_index = i
                    break

            if voice_index is None: 
                error = "Available tracks:"
                for i, track in enumerate(tracks): error += f"\n\t{i}: {track.name}"
                print(f"Could not find voice part {folder} in {file}; {error}")
                broken_files.append(file)
                continue
            main_track = tracks[voice_index]
            
            # Create a new MIDI file with only the meta track
            new_midi = mido.MidiFile(ticks_per_beat=midi_file.ticks_per_beat)
            new_midi.tracks.append(mido.MidiTrack())
            
            # Collect all meta events from all tracks
            meta_events = []
            for track in midi_file.tracks:
                time = 0
                for msg in track:
                    time += msg.time
                    if msg.is_meta:
                        # Copy the message and adjust its time to be absolute
                        new_msg = msg.copy(time=time)
                        meta_events.append(new_msg)
            # Sort meta events by their absolute time
            meta_events.sort(key=lambda msg: msg.time)
            
            # Add all meta events to the new MIDI file in their original order
            last_time = 0
            for msg in meta_events:
                # Adjust the 'time' of the msg to be relative to the last message in new_meta_track
                msg.time -= last_time
                new_midi.tracks[0].append(msg)
                last_time += msg.time

            # Append the isolated main track to the new MIDI file
            new_midi.tracks.append(main_track)
            
            # Save the new MIDI file to the "Isolated" folder
            new_midi.save(os.path.join(isolated_folder_path, file))
        pass
    if len(broken_files) != 0:
        print(f"\nBroken files in {folder}:")
        for bad_file in broken_files: print(f"\t{bad_file}")
        break
    else: print(f"Completed folder {folder} successfully")
