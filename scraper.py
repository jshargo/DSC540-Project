import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO as Str

class StatsScraper:            
    def __init__(self, player_name=""):
        self.player_name = player_name
        self.player_urls = self.load_player_urls()
        self.player_url = self.get_player_url(player_name)
        if self.player_url == "Player not found.":
            raise ValueError(f"Player '{player_name}' not found in player_urls.json")
        # Only slice if the URL ends with '.html'
        if self.player_url.endswith(".html"):
            self.player_url = self.player_url[:-5]
        
    def load_player_urls(self, filename='player_urls.json'):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    
    def get_player_url(self, player_name):
        return self.player_urls.get(player_name, "Player not found.")
        
class GameLogs(StatsScraper):
    def fetch_data(self, seasons=[2024]):
        all_data = []
        for season in seasons:
            logs_website = f"{self.player_url}/gamelog/{season}"
            logs = requests.get(logs_website)
                  
            if logs.status_code == 200:
                soup = BeautifulSoup(logs.text, "html.parser")
                logs_table = soup.find(id="pgl_basic")
                
                # Remove header rows if needed
                thead_elements = soup.find_all('tr', class_="thead")
                for thead in thead_elements:
                    thead.decompose()
                
                if logs_table:
                    season_data = pd.read_html(Str(str(logs_table)))[0]
                    season_data['Season'] = season  # add season info
                    all_data.append(season_data)
                    
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

# Example usage:
# This will fetch game logs for the player for the 2023, 2024, and 2025 seasons.
luka_logs = GameLogs("Luka Dončić")
df = luka_logs.fetch_data(seasons=[2019, 2020, 2021, 2022, 2023, 2024, 2025])
# df = data.to_csv("luka_game_logs.csv", index=False)

# Step 2: Clean the data using your partner's cleaning code
df = df.fillna(0)

# Remove rows where 'GS' is 'Inactive' or 'Did Not Dress'
df = df[~df['GS'].isin(['Inactive', 'Did Not Dress'])]

# Drop unnecessary columns if they exist
columns_to_drop = ['Rk', 'Age', 'Unnamed: 7', 'G', 'ORB', 'DRB', 'GS']
for column in columns_to_drop:
    if column in df.columns:
        df = df.drop(column, axis=1)

# Rename 'Unnamed: 7' to 'W/L' if it exists
if 'Unnamed: 7' in df.columns:
    df = df.rename(columns={'Unnamed: 7': 'W/L'})

# Convert 'Date' to datetime then calculate rest (difference in days)
df['Date'] = pd.to_datetime(df['Date'])
df['Rest'] = df['Date'].diff().dt.days.fillna(0)
df.reset_index(drop=True, inplace=True)

# Map opponent teams to numeric values
team_to_number = {
    'ATL': 1, 'BOS': 2, 'BRK': 3, 'CHI': 4, 'CHO': 5, 'CLE': 6, 'DAL': 7,
    'DEN': 8, 'DET': 9, 'GSW': 10, 'HOU': 11, 'IND': 12, 'LAC': 13, 'LAL': 14,
    'MEM': 15, 'MIA': 16, 'MIL': 17, 'MIN': 18, 'NOP': 19, 'NYK': 20, 'OKC': 21,
    'ORL': 22, 'PHI': 23, 'PHO': 24, 'POR': 25, 'SAC': 26, 'SAS': 27, 'TOR': 28,
    'UTA': 29, 'WAS': 30,
}
df['Opp'] = df['Opp'].apply(lambda x: team_to_number.get(x, x))

# Step 3: Export the cleaned data to a CSV file
df.to_csv("LukaDoncic.csv", index=False)
