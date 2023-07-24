import os
import csv

root_dir = './'

# Create a CSV file
with open('composer_piece.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['composer', 'piece name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    # Scan the directory
    for dir_name in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, dir_name)):
            # Split the directory name into composer and piece name
            parts = dir_name.rsplit(' - ', 1)
            if len(parts) == 2:
                piece_name, composer = parts
                writer.writerow({'composer': composer, 'piece name': piece_name})