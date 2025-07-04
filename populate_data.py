import random
import numpy as np
from multiprocessing import Process, cpu_count
import os
import time
from champion_list import CHAMPION_LIST, ROLE_CHAMPIONS

# === Constants === #
NUM_GAMES = 1000000
ROLES = ["top", "jungle", "mid", "bot", "support"]
NUM_PROCESSES = min(cpu_count(), 8)
OUTPUT_FILE = "league_games.npz"

NUM_CHAMPIONS = len(CHAMPION_LIST)
CHAMPION_TO_IDX = {name.lower(): idx for idx, name in enumerate(CHAMPION_LIST)}
ROLE_TO_IDX = {role: idx for idx, role in enumerate(ROLES)}
PLAYERS_PER_GAME = 10

PLAYER_FEATURES = NUM_CHAMPIONS + len(ROLES) + 6  # 168 + 5 + 6 = 178
MATCH_FEATURES = PLAYER_FEATURES * PLAYERS_PER_GAME # 1780 = 178 * 10

# Ranges for random generation
MIN_MATCH_TIME_SECONDS = 1200  # 20 minutes in seconds
MAX_MATCH_TIME_SECONDS = 3000  # 50 minutes in seconds

# ========== UTILS ==========
def sample_unique_role_champions():
    selected = set()
    role_champions = {}
    for role in ROLES:
        while True:
            champ = random.choice(ROLE_CHAMPIONS[role])
            if champ not in selected:
                selected.add(champ)
                role_champions[role] = champ
                break
    return role_champions

def generate_stats(role):
    if role == "support":
        return {
            "gold": random.randint(4000, 9000),
            "damage_dealt": random.randint(3000, 12000),
            "damage_taken": random.randint(6000, 15000),
            "cs": random.randint(10, 60)
        }
    elif role == "bot":
        return {
            "gold": random.randint(8000, 15000),
            "damage_dealt": random.randint(10000, 25000),
            "damage_taken": random.randint(4000, 12000),
            "cs": random.randint(100, 300)
        }
    else:
        return {
            "gold": random.randint(7000, 14000),
            "damage_dealt": random.randint(8000, 20000),
            "damage_taken": random.randint(6000, 16000),
            "cs": random.randint(80, 240)
        }

def generate_game():
    game = []
    duration = random.randint(MIN_MATCH_TIME_SECONDS, MAX_MATCH_TIME_SECONDS)  # in seconds
    blue_team = sample_unique_role_champions()
    red_team = sample_unique_role_champions()
    while set(blue_team.values()).intersection(set(red_team.values())):
        red_team = sample_unique_role_champions()

    blue_win = random.choice([0, 1])

    for team, champs, win_flag in [("blue", blue_team, blue_win), ("red", red_team, 1 - blue_win)]:
        for role, champ in champs.items():
            stats = generate_stats(role)
            player_data = [
                champ,
                role,
                round(random.uniform(0.5, 5.0), 2),
                stats["gold"],
                stats["damage_dealt"],
                stats["damage_taken"],
                stats["cs"],
                duration,
                team,
                "win" if win_flag else "loss"
            ]
            game.append(player_data)
    return game
def encode_player(player):

    champion, role, kda, gold, dmg_dealt, dmg_taken, cs, time_played, team, outcome = player

    champ_idx = CHAMPION_TO_IDX[champion.lower()]
    champ_onehot = np.zeros(NUM_CHAMPIONS)
    champ_onehot[champ_idx] = 1.0

    role_idx = ROLE_TO_IDX[role]
    role_onehot = np.zeros(len(ROLES))
    role_onehot[role_idx] = 1.0

    stats = np.array([kda, gold, dmg_dealt, dmg_taken, cs, time_played], dtype=np.float32)

    return np.concatenate([champ_onehot, role_onehot, stats])

def game_to_features(game):
    
    blue = [p for p in game if p[8] == "blue"]
    red = [p for p in game if p[8] == "red"]

    blue_sorted = sorted(blue, key=lambda x: ROLE_TO_IDX[x[1]])
    red_sorted = sorted(red, key=lambda x: ROLE_TO_IDX[x[1]])

    X_parts = [encode_player(p) for p in blue_sorted + red_sorted]
    X = np.concatenate(X_parts)
    y = 1.0 if blue[0][-1] == "win" else 0.0

    return X, y

def worker(process_idx, games_per_process):
    X_data = []
    y_data = []

    for _ in range(games_per_process):
        game = generate_game()
        X, y = game_to_features(game)
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