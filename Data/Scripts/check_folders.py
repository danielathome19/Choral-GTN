import os

print("Checking folders for songs with more/less than four files...")
for songfolder in os.listdir():
    if os.path.isdir(songfolder):
        print("\nFolder: " + songfolder)
        for folder in os.listdir(songfolder):
            if os.path.isdir(songfolder + "/" + folder):
                if len(os.listdir(songfolder + "/" + folder)) != 4:
                    print(f"\t{folder}")
