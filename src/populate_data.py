import numpy as np
import os
import time
import json
from multiprocessing import Process, cpu_count

NUM_PROCESSES = max(cpu_count(), 16)
NUM_GAMES = 50000000

OUTPUT_FILE = "league_games.npz"

def parse_game(match_details):
    """
    Parse match details from Riot API into feature vectors and labels for model training
    
    Args:
        match_details: Dictionary containing match data from Riot API
    
    Returns:
        X: Feature vectors for each champion in teams and general statistics 
        y: Label (1 if blue team won, 0 if red team won)
    """
    if not match_details:
        print("Match details are empty")
        return np.zeros(20), 0
    
    # Load champion statistics from league_stats.json
    try:
        with open('league_stats.json', 'r') as f:
            league_stats = json.load(f)
        
        # Create lookup dictionaries for win rate, pick rate, and ban rate by champion name
        win_rates = {}
        pick_rates = {}
        ban_rates = {}
        
        for champ_data in league_stats["CHAMPION_LIST_GLOBAL_WINRATE"]:
            if champ_data["name"] not in win_rates:
                win_rates[champ_data["name"]] = champ_data["winrate"]
        
        for champ_data in league_stats["CHAMPION_LIST_GLOBAL_PICKRATE"]:
            if champ_data["name"] not in pick_rates:
                pick_rates[champ_data["name"]] = champ_data["pickrate"]
        
        for champ_data in league_stats["CHAMPION_LIST_GLOBAL_BANRATE"]:
            if champ_data["name"] not in ban_rates:
                ban_rates[champ_data["name"]] = champ_data["banrate"]
    except Exception as e:
        print(f"Error loading league_stats.json: {e}")
        return np.zeros(20), 0
    
    # Extract team data
    blue_team = next((team for team in match_details.get('teams', []) if team.get('team_id') == 100), {})
    red_team = next((team for team in match_details.get('teams', []) if team.get('team_id') == 200), {})
    
    # Extract participant data by team
    blue_participants = [p for p in match_details.get('participants', []) if p.get('team_id') == 100]
    red_participants = [p for p in match_details.get('participants', []) if p.get('team_id') == 200]
    
    # Create feature vector
    features = []
    
    # Champion features for blue team
    for participant in blue_participants:
        champion_name = participant.get('champion_name', '')
        champion_id = participant.get('champion_id', 0)
        team_id = participant.get('team_id', 0)
        
        # Get champion stats from our lookup dictionaries
        win_rate = win_rates.get(champion_name, 0.5)
        pick_rate = pick_rates.get(champion_name, 0.05)
        ban_rate = ban_rates.get(champion_name, 0.05)
        
        features.extend([
            champion_id,
            win_rate,
            pick_rate,
            ban_rate,
            team_id
        ])
    
    # Champion features for red team
    for participant in red_participants:
        champion_name = participant.get('champion_name', '')
        champion_id = participant.get('champion_id', 0)
        team_id = participant.get('team_id', 0)
        
        # Get champion stats from our lookup dictionaries
        win_rate = win_rates.get(champion_name, 0.5)
        pick_rate = pick_rates.get(champion_name, 0.05)
        ban_rate = ban_rates.get(champion_name, 0.05)
        
        features.extend([
            champion_id,
            win_rate,
            pick_rate,
            ban_rate,
            team_id
        ])
    
    # Create label (1 if blue team won, 0 if red team won)
    label = 1 if blue_team.get('win', False) else 0
    
    # Convert to numpy arrays
    X = np.array(features, dtype=np.float32)
    y = np.array(label, dtype=np.float32)
    
    return X, y

def worker(process_idx, games_per_process):
    X_data = []
    y_data = []

    for _ in range(games_per_process):
        # Create empty match details for testing
        match_details = {}
        X, y = parse_game(match_details)
        X_data.append(X)
        y_data.append(y)

    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)

    np.savez_compressed(f"data_part_{process_idx}.npz", X=X_data, y=y_data)

def merge_npz_files(n_parts, output_file):
    X_all = []
    y_all = []

    for i in range(n_parts):
        with np.load(f"data_part_{i}.npz") as npz:
            X_all.append(npz['X'])
            y_all.append(npz['y'])
        os.remove(f"data_part_{i}.npz")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    np.savez_compressed(output_file, X=X_all, y=y_all)

# ========== MAIN ==========
if __name__ == "__main__":
    start = time.time()
    games_per_process = NUM_GAMES // NUM_PROCESSES
    processes = []

    for i in range(NUM_PROCESSES):
        p = Process(target=worker, args=(i, games_per_process))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merge_npz_files(NUM_PROCESSES, OUTPUT_FILE)
    end = time.time()
    print(f"Generated {NUM_GAMES} games into {OUTPUT_FILE} in {end - start:.2f} seconds using {NUM_PROCESSES} processes.")
