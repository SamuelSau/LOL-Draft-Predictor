import os
import sys
import numpy as np
from datetime import datetime
from src.populate_data import parse_game

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "league_games.npz")
puuids_file_path = os.path.join(PROJECT_ROOT, "list_of_PUUIDS.txt")
match_ids_file_path = os.path.join(PROJECT_ROOT, "list_of_matchIDS.txt")
match_details_file_path = os.path.join(PROJECT_ROOT, "list_of_match_details.txt")

def get_routing_value(region):
    """Convert region code to routing value"""
    region_mappings = {
        "na1": "americas",
        "br1": "americas",
        "la1": "americas",
        "la2": "americas",
        "oc1": "americas",
        "me1": "europe",
        "euw1": "europe",
        "eun1": "europe",
        "tr1": "europe",
        "ru": "europe",
        "kr": "asia",
        "jp1": "asia",
        "ph2": "asia",
        "sg2": "asia",
        "tw2": "asia",
        "vn2": "asia"
    }
    
    if region in region_mappings:
        print(f"region found in region_mappings: {region}")
        return region_mappings[region]
    
    else:
        print(f"Region {region} not found in mappings, defaulting to americas...")
        return "americas"

def load_existing_data(puuids_file_path: str, match_ids_file_path: str, match_details_file_path: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[str]]:
    """Load puuids, match_ids, and match_details if files exist"""
    all_puuids_region = set()
    all_match_ids_region = set()
    all_match_details = set()
    
    if os.path.exists(puuids_file_path) and os.path.getsize(puuids_file_path) > 0:
        with open(puuids_file_path, 'r') as f:
            for line in f:
                puuid, region = line.strip().split(',')
                if puuid and region:
                    all_puuids_region.add((puuid, region))
        print(f"Loaded {len(all_puuids_region)} existing PUUIDs from {puuids_file_path}")
        
    if os.path.exists(match_ids_file_path) and os.path.getsize(match_ids_file_path) > 0:
        with open(match_ids_file_path, 'r') as f:
            for line in f:
                match_id, region = line.strip().split(',')
                if match_id and region:
                    all_match_ids_region.add((match_id, region))
        print(f"Loaded {len(all_match_ids_region)} existing match IDs from {match_ids_file_path}")
    
    if os.path.exists(match_details_file_path) and os.path.getsize(match_details_file_path) > 0: #read from file, do not add into cache since it's too large for overhead
        print(f"Match details file {match_details_file_path} already exists. Skipping match details collection.")
    
    return all_puuids_region, all_match_ids_region, all_match_details

def save_match_details_to_npz(filename=OUTPUT_FILE):
    """Read from match_details file and convert to np obects"""
    match_details_list = []

    if os.path.exists(match_details_file_path) and os.path.getsize(match_details_file_path) > 0:
        with open(match_details_file_path, 'r') as f:
            for line in f:
                match_details = line.strip()
                if match_details:
                    match_details_list.append(match_details)
        print(f"Loaded {len(match_details_list)} match details from {match_details_file_path}")
    
    try:
        X_data = []
        y_data = []
        
        for match_details in match_details_list:
            X, y = parse_game(match_details)
            X_data.append(X)
            y_data.append(y)
        
        X_data = np.array(X_data, dtype=np.float32)
        y_data = np.array(y_data, dtype=np.float32)
        
        if os.path.exists(filename):
            with np.load(filename) as existing_data:
                X_existing = existing_data['X']
                y_existing = existing_data['y']
                X_combined = np.concatenate([X_existing, X_data])
                y_combined = np.concatenate([y_existing, y_data])
            np.savez_compressed(filename, X=X_combined, y=y_combined)
        else:
            np.savez_compressed(filename, X=X_data, y=y_data)
        
        print(f"Saved {len(X_data)} match details to {filename}")
    except Exception as e:
        print(f"Error saving match details: {str(e)}")

def ensure_files_exist():
    """Checks if the puuids, match_ids, and match_details files exist in project root"""
    if not os.path.exists(puuids_file_path):
        os.makedirs(os.path.dirname(puuids_file_path), exist_ok=True)
        with open(puuids_file_path, 'w') as f:
            pass
        print(f"Created empty file: {puuids_file_path}")
    
    if not os.path.exists(match_ids_file_path):
        os.makedirs(os.path.dirname(match_ids_file_path), exist_ok=True)
        with open(match_ids_file_path, 'w') as f:
            pass
        print(f"Created empty file: {match_ids_file_path}")

    if not os.path.exists(match_details_file_path):
        os.makedirs(os.path.dirname(match_details_file_path), exist_ok=True)
        with open(match_details_file_path, 'w') as f:
            pass
        print(f"Created empty file: {match_details_file_path}")

def calculate_time_difference(start_time: datetime, end_time: datetime) -> float:
    """Calculate the time difference in seconds between two datetime objects"""
    if start_time and end_time:
        return (end_time - start_time).total_seconds()
    else:
        return None