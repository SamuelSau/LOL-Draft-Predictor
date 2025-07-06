import json
import os  
#safety check for opening the constants file

if not os.path.exists("league_constants.json"):
    print("Error: league_constants.json not found. Please ensure it exists in the current directory.")
    import sys
    sys.exit(1)

with open('league_constants.json', 'r') as f:
    REGIONS = json.load(f)["REGIONS"]
    print("Regions: ", REGIONS)
    ROLE_CHAMPIONS = json.load(f)["ROLE_CHAMPIONS"]
    print("Role Champions: ", ROLE_CHAMPIONS)

def generate_stats(ROLE_CHAMPIONS, region_prefix=""):
    stats = {
        f"CHAMPION_LIST_{region_prefix}WINRATE": [],
        f"CHAMPION_LIST_{region_prefix}PICKRATE": [],
        f"CHAMPION_LIST_{region_prefix}BANRATE": [],
    }

    for role in ROLE_CHAMPIONS:
        stats[f"CHAMPION_LIST_{region_prefix}WINRATE"].append({
            "name": name,
            "role": role,
            "winrate": round(random.uniform(0.45, 0.60), 2)
        })
        stats[f"CHAMPION_LIST_{region_prefix}PICKRATE"].append({
            "name": name,
            "role": role,
            "pickrate": round(random.uniform(0.05, 0.25), 2)
        })
        stats[f"CHAMPION_LIST_{region_prefix}BANRATE"].append({
            "name": name,
            "role": role,
            "banrate": round(random.uniform(0.05, 0.30), 2)
        })
    
    return stats

# Initialize full data
league_stats = {}

# Add region-specific stats
for region in REGIONS:
    region_stats = generate_stats(ROLE_CHAMPIONS, region_prefix=region.upper() + "_")
    league_stats.update(region_stats)

# Add global stats
global_stats = generate_stats(ROLE_CHAMPIONS, region_prefix="GLOBAL_")
league_stats.update(global_stats)

# Save to JSON
with open("league_stats.json", "w") as f:
    json.dump(league_stats, f, indent=4)

print("league_stats.json generated successfully.")