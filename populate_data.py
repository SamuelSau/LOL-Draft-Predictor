import random
import numpy as np
from multiprocessing import Process, cpu_count
import os
import time
from champion_list import CHAMPION_LIST

# === Constants === #
NUM_GAMES = 100000
ROLES = ["top", "jungle", "mid", "bot", "support"]
NUM_PROCESSES = min(cpu_count(), 8)

NUM_CHAMPIONS = len(CHAMPION_LIST)
CHAMPION_TO_IDX = {name.lower(): idx for idx, name in enumerate(CHAMPION_LIST)}
ROLE_TO_IDX = {role: idx for idx, role in enumerate(ROLES)}

PLAYER_FEATURES = NUM_CHAMPIONS + len(ROLES) + 6  # 167 + 5 + 6 = 178
MATCH_FEATURES = PLAYER_FEATURES * 10             # 10 players per game

# === Game Feature Generator === #
def generate_match_features():
    match_champs = random.sample(CHAMPION_LIST, 10)
    blue_champs = match_champs[:5]
    red_champs = match_champs[5:]
    blue_wins = random.choice([True, False])
    time_seconds = random.randint(20 * 60, 40 * 60)

    match_features = []

    for i, role in enumerate(ROLES):
        # Blue player
        features = generate_player_features(blue_champs[i], role, time_seconds)
        match_features.append(features)
        # Red player
        features = generate_player_features(red_champs[i], role, time_seconds)
        match_features.append(features)

    x = np.concatenate(match_features, dtype=np.float32)
    y = np.array([1 if blue_wins else 0], dtype=np.float32)
    return x, y

def generate_player_features(champ, role, time_played):
    champ_onehot = np.zeros(NUM_CHAMPIONS, dtype=np.float32)
    champ_onehot[CHAMPION_TO_IDX[champ.lower()]] = 1.0

    role_onehot = np.zeros(len(ROLES), dtype=np.float32)
    role_onehot[ROLE_TO_IDX[role]] = 1.0

    kda = round(random.uniform(0.5, 5.0), 2)
    gold = random.randint(5000, 15000)
    dmg_dealt = random.randint(5000, 25000)
    dmg_taken = random.randint(5000, 20000)
    cs = random.randint(30, 300)

    stats = np.array([kda, gold, dmg_dealt, dmg_taken, cs, time_played], dtype=np.float32)
    return np.concatenate([champ_onehot, role_onehot, stats])

# === Worker Function === #
def generate_data_batch(process_id, num_games):
    X_batch = []
    y_batch = []
    for _ in range(num_games):
        x, y = generate_match_features()
        X_batch.append(x)
        y_batch.append(y)

    X_batch = np.array(X_batch, dtype=np.float32)
    y_batch = np.array(y_batch, dtype=np.float32)

    np.savez_compressed(f"data_part_{process_id}.npz", X=X_batch, y=y_batch)
    print(f"Process {process_id}: Saved {num_games} games.")

# === Merge .npz files === #
def merge_npz_files(num_parts, output_file):
    X_all = []
    y_all = []

    for i in range(num_parts):
        with np.load(f"data_part_{i}.npz") as part:
            X_all.append(part['X'])
            y_all.append(part['y'])

        # Now that file is closed, it's safe to delete
        os.remove(f"data_part_{i}.npz")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    np.savez_compressed(output_file, X=X_all, y=y_all)
    print(f"Merged into {output_file} with shape X={X_all.shape}, y={y_all.shape}")

# === Main Execution === #
if __name__ == "__main__":
    start = time.time()

    games_per_proc = NUM_GAMES // NUM_PROCESSES
    processes = []

    for i in range(NUM_PROCESSES):
        p = Process(target=generate_data_batch, args=(i, games_per_proc))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merge_npz_files(NUM_PROCESSES, "example.npz")
    end = time.time()
    print(f"✅ Generated {NUM_GAMES} games into example.npz in {end - start:.2f} seconds.")