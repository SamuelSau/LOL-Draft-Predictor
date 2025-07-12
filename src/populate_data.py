import numpy as np
import os
import time
import json
import sys
from multiprocessing import Process, cpu_count

NUM_PROCESSES = max(cpu_count(), 16)
NUM_GAMES = 50000000

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "league_games.npz")
league_stats_file_path = os.path.join(PROJECT_ROOT, "stats/league_stats.json")

with open(league_stats_file_path, 'r') as f:
        league_stats = json.load(f)

champion_stats = {}

def extract_stat(stat_list, key):
    for entry in stat_list:
        name = entry["name"]
        if name not in champion_stats:
            champion_stats[name] = {}
        champion_stats[name][key] = entry.get(key[:-1] if key.endswith("s") else key)

extract_stat(league_stats["CHAMPION_LIST_GLOBAL_WINRATE"], "global_winrate")
extract_stat(league_stats["CHAMPION_LIST_GLOBAL_PICKRATE"], "global_pickrate")
extract_stat(league_stats["CHAMPION_LIST_GLOBAL_BANRATE"], "global_banrate")

# Sort all champion names to maintain one-hot encoding order
all_champions = sorted(champion_stats.keys())
"""
['Aatrox', 'Ahri', 'Akali', 'Alistar', 'Amumu', 'Anivia', 'Annie', 'Aphelios', 'Ashe', 'Aurelion Sol', 'Aurora', 'Azir', 'Bard', "Bel'Veth", 'Blitzcrank', 'Brand', 'Braum', 'Briar', 'Caitlyn', 'Camille', 'Cassiopeia', "Cho'Gath", 'Corki', 'Darius', 'Diana', 'Dr. Mundo', 'Draven', 'Ekko', 'Elise', 'Evelynn', 'Ezreal', 'Fiddlesticks', 'Fiora', 'Fizz', 'Galio', 'Gangplank', 'Garen', 'Gnar', 'Gragas', 'Graves', 'Gwen', 'Hecarim', 'Heimerdinger', 'Hwei', 'Illaoi', 'Irelia', 'Ivern', 'Janna', 'Jarvan IV', 'Jax', 'Jayce', 'Jhin', 'Jinx', "K'Sante", "Kai'Sa", 'Kalista', 'Karma', 'Karthus', 'Kassadin', 'Katarina', 'Kayle', 'Kayn', 'Kennen', "Kha'Zix", 'Kindred', 'Kled', "Kog'Maw", 'LeBlanc', 'Lee Sin', 'Leona', 'Lillia', 'Lissandra', 'Lucian', 'Lulu', 'Lux', 'Malphite', 'Malzahar', 'Maokai', 'Master Yi', 'Mel', 'Milio', 'Miss Fortune', 'Mordekaiser', 'Morgana', 'Naafiri', 'Nami', 'Nasus', 'Nautilus', 'Neeko', 'Nidalee', 'Nilah', 'Nocturne', 'Nunu & Willump', 'Olaf', 'Orianna', 'Ornn', 'Pantheon', 'Poppy', 'Pyke', 'Qiyana', 'Quinn', 'Rakan', 'Rammus', "Rek'Sai", 'Rell', 'Renata Glasc', 'Renekton', 'Riven', 'Rumble', 'Ryze', 'Samira', 'Sejuani', 'Senna', 'Seraphine', 'Sett', 'Shaco', 'Shen', 'Shyvana', 'Singed', 'Sion', 'Sivir', 'Smolder', 'Sona', 'Soraka', 'Swain', 'Sylas', 'Syndra', 'Tahm Kench', 'Taliyah', 'Talon', 'Taric', 'Teemo', 'Thresh', 'Tristana', 'Trundle', 'Tryndamere', 'Twisted Fate', 'Twitch', 'Udyr', 'Urgot', 'Varus', 'Vayne', "Vel'Koz", 'Vex', 'Vi', 'Viego', 'Viktor', 'Vladimir', 'Volibear', 'Warwick', 'Wukong', 'Xayah', 'Xerath', 'Xin Zhao', 'Yasuo', 'Yone', 'Yorick', 'Yuumi', 'Zac', 'Zed', 'Zeri', 'Ziggs', 'Zilean', 'Zoe', 'Zyra']
"""

def normalize_name(name: str):
    #irelia
    return name.lower().replace(" ", "").replace("'", "").replace(".", "").replace("&", "").replace("-", "")

normalized_to_real_name = { 
    # {"drmundo": "Dr Mundo"}
    normalize_name(real_name): real_name
    for real_name in all_champions
}
"""
{'aatrox': 'Aatrox', 'ahri': 'Ahri', 'akali': 'Akali', 'alistar': 'Alistar', 'amumu': 'Amumu', 'anivia': 'Anivia', 'annie': 'Annie', 'aphelios': 'Aphelios', 'ashe': 'Ashe', 'aurelionsol': 'Aurelion Sol', 'aurora': 'Aurora', 'azir': 'Azir', 'bard': 'Bard', 'belveth': "Bel'Veth", 'blitzcrank': 'Blitzcrank', 'brand': 'Brand', 'braum': 'Braum', 'briar': 'Briar', 'caitlyn': 'Caitlyn', 'camille': 'Camille', 'cassiopeia': 'Cassiopeia', 'chogath': "Cho'Gath", 'corki': 'Corki', 'darius': 'Darius', 'diana': 'Diana', 'drmundo': 'Dr. Mundo', 'draven': 'Draven', 'ekko': 'Ekko', 'elise': 'Elise', 'evelynn': 'Evelynn', 'ezreal': 'Ezreal', 'fiddlesticks': 'Fiddlesticks', 'fiora': 'Fiora', 'fizz': 'Fizz', 'galio': 'Galio', 'gangplank': 'Gangplank', 'garen': 'Garen', 'gnar': 'Gnar', 'gragas': 'Gragas', 'graves': 'Graves', 'gwen': 'Gwen', 'hecarim': 'Hecarim', 'heimerdinger': 'Heimerdinger', 'hwei': 'Hwei', 'illaoi': 'Illaoi', 'irelia': 'Irelia', 'ivern': 'Ivern', 'janna': 'Janna', 'jarvaniv': 'Jarvan IV', 'jax': 'Jax', 'jayce': 'Jayce', 'jhin': 'Jhin', 'jinx': 'Jinx', 'ksante': "K'Sante", 'kaisa': "Kai'Sa", 'kalista': 'Kalista', 'karma': 'Karma', 'karthus': 'Karthus', 'kassadin': 'Kassadin', 'katarina': 'Katarina', 'kayle': 'Kayle', 'kayn': 'Kayn', 'kennen': 'Kennen', 'khazix': "Kha'Zix", 'kindred': 'Kindred', 'kled': 'Kled', 'kogmaw': "Kog'Maw", 'leblanc': 'LeBlanc', 'leesin': 'Lee Sin', 'leona': 'Leona', 'lillia': 'Lillia', 'lissandra': 'Lissandra', 'lucian': 'Lucian', 'lulu': 'Lulu', 'lux': 'Lux', 'malphite': 'Malphite', 'malzahar': 'Malzahar', 'maokai': 'Maokai', 'masteryi': 'Master Yi', 'mel': 'Mel', 'milio': 'Milio', 'missfortune': 'Miss Fortune', 'mordekaiser': 'Mordekaiser', 'morgana': 'Morgana', 'naafiri': 'Naafiri', 'nami': 'Nami', 'nasus': 'Nasus', 'nautilus': 'Nautilus', 'neeko': 'Neeko', 'nidalee': 'Nidalee', 'nilah': 'Nilah', 'nocturne': 'Nocturne', 'nunuwillump': 'Nunu & Willump', 'olaf': 'Olaf', 'orianna': 'Orianna', 'ornn': 'Ornn', 'pantheon': 'Pantheon', 'poppy': 'Poppy', 'pyke': 'Pyke', 'qiyana': 'Qiyana', 'quinn': 'Quinn', 'rakan': 'Rakan', 'rammus': 'Rammus', 'reksai': "Rek'Sai", 'rell': 'Rell', 'renataglasc': 'Renata Glasc', 'renekton': 'Renekton', 'riven': 'Riven', 'rumble': 'Rumble', 'ryze': 'Ryze', 'samira': 'Samira', 'sejuani': 'Sejuani', 'senna': 'Senna', 'seraphine': 'Seraphine', 'sett': 'Sett', 'shaco': 'Shaco', 'shen': 'Shen', 'shyvana': 'Shyvana', 'singed': 'Singed', 'sion': 'Sion', 'sivir': 'Sivir', 'smolder': 'Smolder', 'sona': 'Sona', 'soraka': 'Soraka', 'swain': 'Swain', 'sylas': 'Sylas', 'syndra': 'Syndra', 'tahmkench': 'Tahm Kench', 'taliyah': 'Taliyah', 'talon': 'Talon', 'taric': 'Taric', 'teemo': 'Teemo', 'thresh': 'Thresh', 'tristana': 'Tristana', 'trundle': 'Trundle', 'tryndamere': 'Tryndamere', 'twistedfate': 'Twisted Fate', 'twitch': 'Twitch', 'udyr': 'Udyr', 'urgot': 'Urgot', 'varus': 'Varus', 'vayne': 'Vayne', 'velkoz': "Vel'Koz", 'vex': 'Vex', 'vi': 'Vi', 'viego': 'Viego', 'viktor': 'Viktor', 'vladimir': 'Vladimir', 'volibear': 'Volibear', 'warwick': 'Warwick', 'wukong': 'Wukong', 'xayah': 'Xayah', 'xerath': 'Xerath', 'xinzhao': 'Xin Zhao', 'yasuo': 'Yasuo', 'yone': 'Yone', 'yorick': 'Yorick', 'yuumi': 'Yuumi', 'zac': 'Zac', 'zed': 'Zed', 'zeri': 'Zeri', 'ziggs': 'Ziggs', 'zilean': 'Zilean', 'zoe': 'Zoe', 'zyra': 'Zyra'}
"""

normalized_all_champions_list = [
    normalize_name(name)
    for name in all_champions
]

def match_to_feature_vector(match_details):
    print(match_details)
    one_hot_vector = np.zeros(len(all_champions) * 2)  # 10 champs (5 per team), so we mark them
    stats_vector = []

    for participant in match_details["participants"]:
        raw_name = participant["champion_name"]
        normalized_name = normalize_name(raw_name)
        team_offset = 0 if participant["team_id"] == 100 else len(all_champions)
        
        if normalized_name in normalized_to_real_name:
            print(f"{normalized_name} is inside normalized_to_real_name")
            champ_index = normalized_all_champions_list.index(normalized_name)
            one_hot_vector[team_offset + champ_index] = 1

        else:
           print(f"Champion '{normalized_name}' not found in normalized_to_real_name dictionary.")

        stats = champion_stats.get(raw_name, {
            "global_winrate": 0.5,
            "global_pickrate": 0.01,
            "global_banrate": 0.01,
        })
        stats_vector.extend([
            stats["global_winrate"],
            stats["global_pickrate"],
            stats["global_banrate"],
        ])

    label = 1 if any(team["team_id"] == 100 and team["win"] for team in match_details["teams"]) else 0
   
    return np.concatenate([one_hot_vector, np.array(stats_vector)]), label

def worker(process_idx, games_per_process):
    X_data = []
    y_data = []

    for _ in range(games_per_process):
        # Create empty match details for testing
        match_details = {}
        X, y = match_to_feature_vector(match_details)
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
