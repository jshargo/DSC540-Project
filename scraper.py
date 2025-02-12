import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO as Str

class StatsScraper:            
    def __init__(self, player_name = ""):
        self.player_name = player_name
        self.player_urls = self.load_player_urls()
        self.player_url = self.get_player_url(player_name)
        self.player_url = self.player_url[:-5]
        
    def load_player_urls(self, filename = 'player_urls.json'):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    
    def get_player_url(self, player_name):
        return self.player_urls.get(player_name, "Player not found.")
    
     #Writes to /players folder HTML code of webpage (If needed)
    '''with open("players/{}.html".format(player_name), "w+") as f:
                f.write(logs.text)'''
        
    
class GameLogs(StatsScraper):
    def fetch_data(self):
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)  
        pd.set_option('display.width', None)
  
        logs_website = f"{self.player_url}/gamelog/2025"
        logs = requests.get(logs_website)
                  
        if logs.status_code == 200:           
            soup = BeautifulSoup(logs.text, "html.parser")
            logs_table = soup.find(id="pgl_basic")
            
            thead_elements = soup.find_all('tr', class_="thead")
            for thead in thead_elements:
                thead.decompose()
     
            if logs_table:
                logs_stats = pd.read_html(Str(str(logs_table)))[0]
                return logs_stats
        

class Splits(StatsScraper):
    def fetch_data(self):
           
        pd.set_option('display.max_columns', None)  
        pd.set_option('display.max_rows', None)       
        splits_website = f"{self.player_url}/splits/2024"
        splits = requests.get(splits_website)
        
        if splits.status_code == 200:           
            soup = BeautifulSoup(splits.text, "html.parser")
            
            over_header = soup.find('tr', class_="over_header")
            if over_header:
                over_header.decompose()
                
            thead_elements = soup.find_all('tr', class_="thead")
            for thead in thead_elements:
                thead.decompose()
            
            splits_table = soup.find(id="splits")
            if splits_table: 
                splits_stats = pd.read_html(Str(str(splits_table)))[0]
                return splits_stats
                  