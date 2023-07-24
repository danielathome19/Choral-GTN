import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def sanitize_folder_name(name):
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(char, '_')
    return name

url = "https://www.choraltech.us/inventory.htm"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table', {'id': 'CTInv'})

last_composer_name = None
last_set_name = None
for row in table.find_all('tr'):
    cells = row.find_all('td')
    if len(cells) == 10:
        composer, piece, _, _, sop, alt, ten, bas, _, _ = cells
        if composer.text.strip():
            if ', ' in composer.text:
                last_name, first_name = composer.text.strip().split(', ')
                last_composer_name = f"{first_name} {last_name}"
            else:
                last_composer_name = composer.text.strip()
        composer_name = last_composer_name
        if sop.text == 'sop' and alt.text == 'alt' and ten.text == 'ten' and bas.text == 'bas':
            piece_name = piece.text.strip()
            if last_set_name and not last_set_name.endswith('(back to top)'):
                folder_name = f"{piece_name} ({last_set_name}) - {composer_name}"
            else:
                folder_name = f"{piece_name} - {composer_name}"
            folder_name = sanitize_folder_name(folder_name)
            print(f"Scraping song: {folder_name}")
            os.makedirs(folder_name, exist_ok=True)
            
            for part, letter in zip([sop, alt, ten, bas], ['S', 'A', 'T', 'B']):
                filename = None
                try:
                    midi_url = urljoin(url, part.find('a')['href'])
                    midi_response = requests.get(midi_url)
                    filename = os.path.join(folder_name, os.path.basename(midi_url).replace('.mid', f'{letter}.mid'))
                    with open(filename, 'wb') as f:
                        f.write(midi_response.content)
                except Exception as e:
                    print(f"Failed to download or save file {filename}. Error: {str(e)}")
    elif len(cells) == 1 and cells[0].get('colspan') == '8' and not cells[0].text.strip().endswith('(back to top)'):
        last_set_name = cells[0].text.strip()