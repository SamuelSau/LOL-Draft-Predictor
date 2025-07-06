import requests
import os
from dotenv import load_dotenv
import numpy as np
from multiprocessing import Process, cpu_count
import json
from populate_data import parse_game, game_to_features, save_match_details_to_npz

load_dotenv()
api_key = os.getenv("API_KEY")

TOTAL_PLAYER_COUNT = 6000
NUM_PROCESSES = max(cpu_count(), 16)
OUTPUT_FILE = "league_games.npz"

#need to test if they work:
with open('league_constants.json', 'r') as f:
    REGIONS = json.load(f)["REGIONS"]
    HIGH_ELO_TIERS = json.load(f)["HIGH_ELO_TIERS"]
    DIVSIONS = json.load(f)["DIVSIONS"]

print("Regions: ", REGIONS)
print("High ELO Tiers: ", HIGH_ELO_TIERS)
print("Divisions: ", DIVSIONS)

def retrieve_top_players_and_puuids(api_key, region, tier, division, start=0, page_count=55):
    try:
        for page_index in range(start, start + page_count):
            url = f"https://{region}.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/{tier}/{division}?page={page_index}&api_key={api_key}"
            headers = {
                "X-Riot-Token": api_key
            }
    except:
        print(f"Error: Invalid region '{region}' or tier '{tier}' or division '{division}'.")
        return []
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        puuid = response.json()[0]["puuid"]
        
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

def retrieve_most_recent_matches(api_key, region, puuid, start=0, count=55):
    
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start={start}&count={count}&api_key={api_key}"
    headers = {
        "X-Riot-Token": api_key
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:

        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

def retrieve_match_details(api_key, match_id):
    url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key=RGAPI-80435652-d446-4ca2-a951-ef280e4c1741"
    headers = {
        "X-Riot-Token": api_key
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()

        #return match details such as champion, position, team, win/loss

    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}

#format should be: champion,position,tier,region,gold,dmg_dealt,dmg_taken,KDA,CS,total_exp,time_played,team,outcome
def save_match_details_to_npz(match_details, filename="league_games.npz"):
    if not os.path.exists(filename):
        with open(filename, 'wb') as f:
            np.savez(f, match_details=match_details)
    else:
        with open(filename, 'ab') as f:
            np.savez(f, match_details=match_details)
    
    print(f"Match details saved to {filename}")
    return None

if __name__ == "__main__":
    if not api_key:
        print("API key not found. Set the API_KEY environment variable.")
    else:
        print("Retrieving top players...")
        for region in REGIONS:
            for tier in HIGH_ELO_TIERS:
                for division in DIVSIONS:
                    retrieve_top_players_and_puuids(api_key, region, tier, division)
                    with open('list_of_PUUIDS.txt', 'r') as f:
                        PUUIDS = [line.strip() for line in f.readlines()]
                    for puuid in PUUIDS:
                        retrieve_most_recent_matches(api_key, region, puuid)
                        with open('list_of_matchIDS.txt', 'r') as f:
                            MATCH_IDS = [line.strip() for line in f.readlines()]
                        for match_id in MATCH_IDS:
                            retrieve_match_details(api_key, match_id)

