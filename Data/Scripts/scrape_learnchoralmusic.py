import re
import os
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

options = Options()
options.add_argument('-headless')
options.binary = FirefoxBinary(r"C:\Users\actes\AppData\Local\Mozilla Firefox\firefox.exe")
driver = webdriver.Firefox(options=options)
driver.get("http://www.learnchoralmusic.co.uk/complist.html")

def sanitize_folder_name(name):
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(char, '')
    return name


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            raise Exception(f"Failed to download file {filename} from {url}")

main_page = driver.current_window_handle

bsoup = BeautifulSoup(driver.page_source, 'html.parser')
table = bsoup.find('table', cellpadding="8")
rows = table.find_all('tr')

for row in rows:
    tds = row.find_all('td')
    composer = tds[0].text
    songs = tds[1].find_all('a')

    for song in songs:
        song_name = song.text
        song_url = 'http://www.learnchoralmusic.co.uk/' + song.get('href')  # [1:]

        if song.find_next_sibling('font'):
            if 'red' in song.find_next_sibling('font').get('color').lower():
                continue

        driver.execute_script(f'''window.open("{song_url}","_blank");''')
        driver.switch_to.window(driver.window_handles[-1])

        time.sleep(2)

        song_page_soup = BeautifulSoup(driver.page_source, 'html.parser')
        full_composer_name = song_page_soup.find('h2', align='CENTER').find('i').text.strip()
        # If the composer name contains "- arranged by ..." then remove
        if "arranged" in full_composer_name.lower():
            full_composer_name = full_composer_name[:full_composer_name.lower().index(" - arranged")].strip()
        try: song_table = song_page_soup.find('tbody')
        except:
            print(f"Failed to find song table for URL: {song_url}; skipping...")
            driver.close()
            driver.switch_to.window(main_page)
            continue

        try: full_song_name = song_page_soup.find('h2', align='LEFT').find('i').text.strip()
        except:
            try: # If the song name is not in the h2 tag, it is in the first row of the table
                full_song_name = song_table.find('tr').find('td').text.strip()
            except:
                print(f"Failed to find song name for URL: {song_url}; skipping...")
                driver.close()
                driver.switch_to.window(main_page)
                continue

        if len(song_table.find_all('tr')) > 3:
            driver.close()
            driver.switch_to.window(main_page)
            continue

        voice_parts = song_table.find_all('th')

        # Check if any of the voice parts have a number in it
        if any([re.search(r'\d', x.text) for x in voice_parts]):
            driver.close()
            driver.switch_to.window(main_page)
            continue
        
        try: midi_links = song_table.find_all('tr')[2].find_all('td')[1:]
        except:
            print(f"Failed to find MIDI links for URL: {song_url}; skipping...")
            driver.close()
            driver.switch_to.window(main_page)
            continue

        # Check if MIDI links are available for all voice parts
        if any([midi.find('a') is None for midi in midi_links]):
            driver.close()
            driver.switch_to.window(main_page)
            continue
        
        try:
            print(f"Scraping song: {song_name}")
            folder_name = sanitize_folder_name(f"{full_song_name} - {full_composer_name}")
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            voice_parts = ['S', 'A', 'T', 'B']
            for i, midi in enumerate(midi_links):
                midi_url = song_url[:song_url.rindex('/')+1] + midi.find('a').get('href').replace('%20', '_')
                midi_name = os.path.basename(midi_url
                                             .replace('.MID', '_' + voice_parts[i] + '.MID')
                                             .replace('.mid', '_' + voice_parts[i] + '.mid'))
                download_file(midi_url, os.path.join(folder_name, midi_name))
        except Exception as e:
                print(f"Failed to scrape song {folder_name}. Error: {str(e)}")

        driver.close()
        driver.switch_to.window(main_page)

driver.quit()
