import os
import sys
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
match_details_file_path = os.path.join(PROJECT_ROOT, "list_of_match_details.txt")
plots_dir = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(plots_dir, exist_ok=True)


def plot_bar(statistics, title, ylabel, filename):
    items = sorted(statistics.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color="skyblue", edgecolor="black")
    plt.title(title, fontsize=14)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01, f"{value:.2f}",
                 ha='center', va='bottom', fontsize=9)

    plt.savefig(os.path.join(plots_dir, f"{filename}.png"))
    plt.close()


def convert_match_id_to_region(match_id):
    return match_id.split('_')[0].lower() if isinstance(match_id, str) and '_' in match_id else None


def calculate_match_statistics(match_details_file=match_details_file_path):
    all_details = {
        "total_matches": 0,
        "number_of_game_versions": set(),
        "matches_per_region": {},
        "win_rate_champions": {},
        "win_rate_by_role": {},
        "win_rate_by_team": {},
    }
    matchId = set()

    with open(match_details_file, 'r') as file:
        for line in file:
            try:
                match_data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line}")
                continue

            match_id = match_data.get("match_id")
            if match_id not in matchId:
                matchId.add(match_id)

                game_version = match_data.get("game_version", "")
                region = convert_match_id_to_region(match_id)
                if not region:
                    continue

                all_details["total_matches"] += 1
                all_details["number_of_game_versions"].add(game_version)
                all_details["matches_per_region"][region] = all_details["matches_per_region"].get(region, 0) + 1

                for participant in match_data.get("participants", []):
                    champion = participant["champion_name"]
                    role = participant["team_position"]
                    team_id = participant["team_id"]
                    win = participant["win"]

                    # Champion win stats
                    if champion not in all_details["win_rate_champions"]:
                        all_details["win_rate_champions"][champion] = {"wins": 0, "total": 0}
                    all_details["win_rate_champions"][champion]["total"] += 1
                    if win:
                        all_details["win_rate_champions"][champion]["wins"] += 1

                    # Role win stats
                    valid_roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
                    if role in valid_roles:
                        if role not in all_details["win_rate_by_role"]:
                            all_details["win_rate_by_role"][role] = {"wins": 0, "total": 0}
                        all_details["win_rate_by_role"][role]["total"] += 1
                        if win:
                            all_details["win_rate_by_role"][role]["wins"] += 1

                    # Team win stats
                    if team_id not in all_details["win_rate_by_team"]:
                        all_details["win_rate_by_team"][team_id] = {"wins": 0, "total": 0}
                    all_details["win_rate_by_team"][team_id]["total"] += 1
                    if win:
                        all_details["win_rate_by_team"][team_id]["wins"] += 1

        return all_details


def plot_all_statistics(statistics):
    # Plot matches per region
    plot_bar(statistics["matches_per_region"], "Matches per Region", "Match Count", "matches_per_region")

    champs_most_played = {
        k: v["total"] if v["total"] > 0 else 0
        for k, v in statistics["win_rate_champions"].items()
    }
    top_champs_played = dict(sorted(champs_most_played.items(), key=lambda x: x[1], reverse=True)[:20])
    plot_bar(top_champs_played, "Top 20 Champion Played", "Number of games played", "most_champs_played")

    # Win rate by champion (only top 20 for readability)
    champ_winrate = {
        k: (v["wins"] / v["total"]) * 100 if v["total"] > 0 else 0
        for k, v in statistics["win_rate_champions"].items()
    }
    top_champs_win_rate = dict(sorted(champ_winrate.items(), key=lambda x: x[1], reverse=True)[:20])
    plot_bar(top_champs_win_rate, "Top 20 Champion Win Rates", "Win Rate (%)", "win_rate_by_champion")
    
    role_labels = {
        "TOP": "TOP",
        "JUNGLE": "JUNGLE",
        "MIDDLE": "MIDDLE",
        "BOTTOM": "ADC",
        "UTILITY": "SUPPORT"
    }
    
    renamed_role_winrate = {
        role_labels.get(role, role): (data["wins"] / data["total"]) * 100 if data["total"] > 0 else 0
        for role, data in statistics["win_rate_by_role"].items()
    }

    plot_bar(renamed_role_winrate, "Win Rate by Role", "Win Rate (%)", "win_rate_by_role")

    most_roles_played = {
        role_labels.get(role, role): data["total"] if data["total"] > 0 else 0
        for role, data in statistics["win_rate_by_role"].items()
    }

    plot_bar(most_roles_played, "Most roles played", "Total games played", "most_roles_played")

    team_labels = {
        "100": "Blue Team",
        "200": "Red Team"
    }

    # Win rate by team
    team_winrate = {
        team_labels.get(str(k), str(k)): (v["wins"] / v["total"]) * 100 if v["total"] > 0 else 0
        for k, v in statistics["win_rate_by_team"].items()
    }

    plot_bar(team_winrate, "Win Rate by Team", "Win Rate (%)", "win_rate_by_team")


if __name__ == "__main__":
    stats = calculate_match_statistics()
    plot_all_statistics(stats)
    print(f"Processed {stats['total_matches']} matches across {len(stats['number_of_game_versions'])} game versions.")
    
