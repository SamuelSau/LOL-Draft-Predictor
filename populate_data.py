import numpy as np
import os
import time
from multiprocessing import Process, cpu_count

NUM_PROCESSES = max(cpu_count(), 16)
NUM_GAMES = 50000000

OUTPUT_FILE = "league_games.npz"

def parse_game(game_data):
    # Placeholder for actual game parsing logic
    # This should convert game data into a format suitable for model training
    X = np.random.rand(10)  # Example feature vector
    y = np.random.randint(0, 2)  # Example label (win/loss)
    return X, y

def game_to_features():
    # Placeholder for actual game-to-features conversion logic
    # This should extract features from the game data
    game_data = {}  # Replace with actual game data retrieval
    X, y = parse_game(game_data)
    return X, y

def worker(process_idx, games_per_process):
    X_data = []
    y_data = []

    for _ in range(games_per_process):
        game = parse_game()
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