import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

def sanitize_folder_name(name):
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(char, '')
    return name

def sanitize_composer_name(name):
    if '"' in name: 
        return name[:name.index('"')].strip()
    elif '(' in name:
        return name[:name.index('(')].strip()

url = "https://www.uma.es/victoria/varios.html"

options = Options()
options.binary = FirefoxBinary(r"C:\Users\actes\AppData\Local\Mozilla Firefox\firefox.exe")
driver = webdriver.Firefox(options=options)

driver.get(url)
# driver.maximize_window()
driver.implicitly_wait(3)

tables = driver.find_elements(By.TAG_NAME, 'table')
tablecount = 0
for table in tables:
    print(f"\nScraping table {tablecount+1}/{len(tables)}")
    rows = table.find_elements(By.TAG_NAME, 'tr')
    last_composer_name = ""
    for row in rows:
        data = row.find_elements(By.TAG_NAME, 'td')
        if len(data) == 3:
            last_composer_name = data[1].text.replace('[?]', '').strip()
            last_composer_name = sanitize_composer_name(last_composer_name)
        elif len(data) == 9:
            try:
                song_name = data[0].find_element(By.TAG_NAME, 'a').text if data[0].find_element(By.TAG_NAME, 'a') else data[0].text
            except: song_name = data[0].text
            song_name.strip()
            try:
                if data[2].text == ' ' or data[2].text == '':
                    composer_name = last_composer_name
                else:
                    composer_name = data[2].text.replace('[?]', '').strip()
                    composer_name = sanitize_composer_name(composer_name)
                voices = row.find_elements(By.CSS_SELECTOR, 'td.voces')
                if any(['\n' in voice.text for voice in voices]) or any([voice.text == ' ' or voice.text == '' for voice in voices]):
                    continue
                midi_links = [voice.find_element(By.TAG_NAME, 'a').get_attribute('href') if voice.find_element(By.TAG_NAME, 'a') else None for voice in voices]
                if all(midi_links):
                    folder_name = sanitize_folder_name(f"{song_name} - {composer_name}")
                    print(f"Scraping song: {folder_name}")
                    os.makedirs(folder_name, exist_ok=True)
                    for i, midi_url in enumerate(midi_links):
                        filename = None
                        midi_url = str(midi_url)
                        try:
                            letter = ['S', 'A', 'T', 'B'][i]
                            midi_response = requests.get(midi_url)
                            filename = os.path.join(folder_name, os.path.basename(midi_url).replace('.mid', f'{letter}.mid'))
                            with open(filename, 'wb') as f:
                                f.write(midi_response.content)
                        except Exception as e:
                            print(f"Failed to download or save file {filename}. Error: {str(e)}")
            except Exception as e:
                print(f"Failed to scrape song {song_name}. Error: {str(e)}")
    tablecount += 1
driver.quit()
