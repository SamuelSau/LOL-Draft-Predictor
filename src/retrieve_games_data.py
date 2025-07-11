import requests
import os
import sys
from dotenv import load_dotenv
import json
import time
from src.utils import get_routing_value, ensure_files_exist, save_match_details_to_npz, load_existing_data, calculate_time_difference

# Add parent directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
api_key = os.getenv("API_KEY")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "league_games.npz")
RATE_LIMIT_WAIT_TIME = 0.05 # Rate limit is 20 requests per second (API limit)
RETRIEVE_RECENT_MATCHES_COUNT = 5
MAX_PAGE_COUNT = 50

match_ids_file_path = os.path.join(PROJECT_ROOT, "list_of_matchIDS.txt")
puuids_file_path = os.path.join(PROJECT_ROOT, "list_of_PUUIDS.txt")
match_details_file_path = os.path.join(PROJECT_ROOT, "list_of_match_details.txt")
league_constants_file_path = os.path.join(PROJECT_ROOT, "constants", "league_constants.json")

try:
    with open(league_constants_file_path, 'r') as f:
        constants = json.load(f)
        REGIONS = list(constants["REGIONS"].keys())
        HIGH_ELO_TIERS = constants["HIGH_ELO_TIERS"]
        DIVISIONS = constants["DIVISIONS"]
except FileNotFoundError:
    print(f"Error: Could not find {league_constants_file_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root directory: {PROJECT_ROOT}")
    print("Please make sure the constants directory exists and contains league_constants.json")
    sys.exit(1)

def collect_puuids_and_write_to_file(api_key: str, puuids_file_path: str, all_puuids_region: set[tuple[str, str]]) -> set[tuple[str, str]]:
    start_time = time.time()
    league_tiers = ["challengerleagues","grandmasterleagues", "masterleagues"]
    headers = {"X-Riot-Token": api_key}

    for region in REGIONS:
        for league_tier in league_tiers:
            try:
                url = f"https://{region}.api.riotgames.com/lol/league/v4/{league_tier}/by-queue/RANKED_SOLO_5x5?api_key={api_key}"
                response = requests.get(url, headers=headers, timeout=10)  
                
                if response.status_code == 200:
                    leagues = response.json()
                    if leagues:
                        players = leagues.get("entries", [])
                        if players:
                            for player in players:
                                puuid = player.get("puuid")
                                if puuid and (puuid, region) not in all_puuids_region:
                                    all_puuids_region.add((puuid, region))
                                    print(f"Added PUUID: {puuid} from {region}/{league_tier}")
                else:
                    print(f"Error retrieving {league_tier} for {region}: {response.status_code} - {response.text}")
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 1))
                        print(f"Rate limited. Waiting for {retry_after} seconds to retry")
                        time.sleep(retry_after)

            except Exception as e:
                print(f"Error retrieving {league_tier} for {region}: {str(e)}")
                continue

    with open(puuids_file_path, 'w') as f:
        for puuid, region in all_puuids_region:
            f.write(f"{puuid},{region}\n")

    end_time = time.time()
    print(f"It took {calculate_time_difference(start_time, end_time)} seconds to collect PUUIDs from all regions...\n")
    return all_puuids_region

def collect_match_ids_and_write_to_file(api_key: str, match_ids_file_path: str, all_puuids_region: set[tuple[str, str]], all_match_ids_region: set[tuple[str, str]]) -> set[tuple[str, str]]:
    
    matchIds = set() #isolate this from the regions since we can temporarily check
    start_time = time.time()

    for puuid, region in all_puuids_region:

            region_routing = get_routing_value(region)
                
            try:
                url = f"https://{region_routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&start=0&count={RETRIEVE_RECENT_MATCHES_COUNT}&api_key={api_key}"
                headers = {"X-Riot-Token": api_key}
                    
                response = requests.get(url, headers=headers)
                    
                if response.status_code == 200:
                    top_recent_match_ids = response.json()
                    print(f"PUUID: {puuid[:10]} Region: {region}\n")

                    if not top_recent_match_ids: #if json returns empty, we exit early to prevent processing further
                        print(f"No recent matches found for PUUID {puuid[:10]} in region {region}")
                        print("--------------------------------\n")
                        continue
                    
                    for current_match_id in top_recent_match_ids:
                        if current_match_id not in matchIds:
                            all_match_ids_region.add((current_match_id, region))
                            matchIds.add(current_match_id)
                            print(f"Added unique match ID: {current_match_id} to all_match_ids_region and matchIds")
                        else:
                            print(f"Already added {current_match_id} in matchIds...")

                    print("--------------------------------")
                    time.sleep(RATE_LIMIT_WAIT_TIME)

                else:
                    print(f"Error retrieving matches for PUUID {puuid}: \n{response.status_code} - {response.text}")
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 1))
                        print(f"Rate limited. Waiting for {retry_after} seconds to retry")
                        time.sleep(retry_after)

            except Exception as e:
                print(f"Error retrieving matches for PUUID {puuid}: {str(e)}")

    # Save collected match IDs to file
    with open(match_ids_file_path, 'w') as f:
        for match_id, region in all_match_ids_region:
            f.write(f"{match_id},{region}\n")
    
    print(f"Collected and saved {len(all_match_ids_region)} unique match IDs")
    del matchIds #empty out the temporary storage of matchIds
    end_time = time.time()
    print(f"It took {calculate_time_difference(start_time, end_time)} seconds to collect match IDs from all PUUIDs in all regions...\n")
    return all_match_ids_region

def collect_match_details(api_key: str, region: str, all_match_ids_region: set[tuple[str, str]]) -> None:
    start_time = time.time()
    region_routing = get_routing_value(region)

    for match_id, region in all_match_ids_region:
        try:
            url = f"https://{region_routing}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={api_key}"
            headers = {"X-Riot-Token": api_key}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                match_data = response.json()
                
                # Extract relevant match details
                processed_data = {
                    'match_id': match_id,
                    'game_version': match_data.get('info', {}).get('gameVersion', ''),
                    'teams': [],
                    'participants': []
                }
                
                # Process team data
                for team in match_data.get('info', {}).get('teams', []):
                    team_data = {
                        'team_id': team.get('teamId', 0),
                        'win': team.get('win', False)
                    }
                    processed_data['teams'].append(team_data)
                
                # Process participant data
                for participant in match_data.get('info', {}).get('participants', []):
                    participant_data = {
                        'champion_id': participant.get('championId', 0),
                        'champion_name': participant.get('championName', ''),
                        'team_id': participant.get('teamId', 0),
                        'team_position': participant.get('teamPosition', ''),
                        'win': participant.get('win', False)
                    }
                    processed_data['participants'].append(participant_data)
                
                #write into file instead due to too much data
                with open(match_details_file_path, 'a') as f:
                    f.write(json.dumps(processed_data) + '\n')

                print(f"Successfully processed match details for match ID: {match_id} in region: {region}")
                time.sleep(RATE_LIMIT_WAIT_TIME)

            else:
                print(f"Error retrieving match details for {match_id}: \n{response.status_code} -- {response.text}")

                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    print(f"Rate limited. Waiting for {retry_after} seconds to retry")
                    time.sleep(retry_after)

                else:
                    print(f"{response.status_code} error: Match ID {match_id} not found in region: {region} in region routing: {region_routing}\n")
                    #for specifically 404 errors:
                
                """
                See in notes.txt for more info
                """
                print("--------------------------------\n")

        except Exception as e:
            print(f"Error processing match ID {match_id}: {str(e)}")

    end_time = time.time()
    print(f"It took {calculate_time_difference(start_time, end_time)} seconds to collect match details for all match IDs in region {region}...\n")
    return None

def integrated_data_collection(api_key: str, puuids_file_path: str, match_ids_file_path: str, match_details_file_path) -> None:
    start_time = time.time()

    all_puuids_region, all_match_ids_region, all_match_details = load_existing_data(puuids_file_path, match_ids_file_path, match_details_file_path)
    # Step 1: Collect PUUIDs if we don't have any
    if not all_puuids_region:
        print("No existing PUUIDs found. Collecting PUUIDs from top players...")
        all_puuids_region = collect_puuids_and_write_to_file(api_key, puuids_file_path, all_puuids_region)
        
    # Step 2: Collect match IDs for each PUUID
    if not all_match_ids_region and all_puuids_region:
        print("Collecting match IDs for PUUIDs...")
        all_match_ids_region = collect_match_ids_and_write_to_file(api_key, match_ids_file_path, all_puuids_region, all_match_ids_region)
    
    # Step 3: Collect match details for each match ID
    if not all_match_details:
        print("No existing match details found. Collecting match details for match IDs...")
        for _, region in all_match_ids_region:
            collect_match_details(api_key, region, all_match_ids_region) #write to file instead of storing data into memory
    
    end_time = time.time()
    print(f"Data collection completed in {calculate_time_difference(start_time, end_time)} seconds")
    return None

if __name__ == "__main__":
    if not api_key:
        print("API key not found. Set the API_KEY environment variable.")

    else:
        start_time = time.time()
        ensure_files_exist()
        integrated_data_collection(api_key, puuids_file_path, match_ids_file_path, match_details_file_path)
        save_match_details_to_npz()
        end_time = time.time()
        print(f"Total time taken for collecting data and saving into npz: {calculate_time_difference(start_time, end_time)} seconds")