# For every folder in the current directory (except the one named "VoiceParts"), 
# iterate through each subfolder in the folder.
# In each subfolder is a MIDI file corresponding to each of the four voice parts, with the last character of the
# filename (before the '.' and the extension) being the voice part (S, A, T, B).
# Grab each of these files and put them in the VoiceParts folder, inside the corresponding subfolder
# (e.g., abc123_S.mid goes in the "Soprano" folder of the VoiceParts folder, and ABC456T.MID goes in the "Tenor" folder)

import os
import shutil

current_dir = os.path.join(os.getcwd(), "Data", "MIDI")
voice_parts_dir = os.path.join(current_dir, "VoiceParts")
count = 1
for folder in os.listdir(current_dir):
    if folder != "VoiceParts":
        folder_path = os.path.join(current_dir, folder)
        for subfolder in os.listdir(folder_path):
            for file in os.listdir(os.path.join(folder_path, subfolder)):
                file_path = os.path.join(folder_path, subfolder, file)
                if file.split('.')[1].lower().strip() != "mid" and file.split('.')[1].lower().strip() != "midi":
                    continue
                voice_part = file.split('.')[0][-1]
                voice_part = 'Soprano' if voice_part == 'S' else 'Alto' if voice_part == 'A' else 'Tenor' if voice_part == 'T' else 'Bass'
                voice_part_path = os.path.join(voice_parts_dir, voice_part)
                if not os.path.exists(voice_part_path):
                    os.mkdir(voice_part_path)
                voice_part_file_path = os.path.join(voice_part_path, str(count) + "_" + file)
                print(voice_part_file_path)
                shutil.copyfile(file_path, voice_part_file_path)
            count += 1
