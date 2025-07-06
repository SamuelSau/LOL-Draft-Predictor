import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os

os.makedirs("plots", exist_ok=True)

# Player counts per tier (as of 7/5/2025 – NA Solo/Duo S13)
tiers = [
    "Challenger", "Grandmaster", "Master", "Diamond",
    "Emerald", "Platinum", "Gold", "Silver", "Bronze", "Iron"
]
counts = np.array([
    300,     # Challenger
    700,     # Grandmaster
    5941,    # Master
    31280,   # Diamond
    118805,  # Emerald
    169972,  # Platinum
    258739,  # Gold
    271214,  # Silver
    241811,  # Bronze
    204239   # Iron
])

# ========= Bar Chart ========= #
plt.figure(figsize=(12, 6))
bars = plt.bar(tiers, counts, color="skyblue", edgecolor="black")
plt.title("NA Season 13 Solo/Duo Player Count by Tier – July 5, 2025", fontsize=14)
plt.xlabel("Tier", fontsize=12)
plt.ylabel("Number of Players", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.gca().set_yticklabels(['{:,}'.format(int(y)) for y in plt.gca().get_yticks()])
plt.tight_layout()

for bar, c in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, f"{c:,}",
             ha='center', va='bottom', fontsize=9)

plt.savefig("plots/tier_bar_chart.png")
plt.close()

# ========= Pie (Donut) Chart ========= #
sorted_data = sorted(zip(counts, tiers), reverse=True)
sorted_counts, sorted_tiers = zip(*sorted_data)

colors = cm.tab20c(np.linspace(0, 1, len(sorted_tiers)))

fig, ax = plt.subplots(figsize=(10, 10))
wedges, texts = ax.pie(
    sorted_counts,
    startangle=140,
    colors=colors,
    wedgeprops=dict(width=0.4)
)

total = sum(sorted_counts)
legend_labels = [
    f"{tier}: {count:,} ({count / total * 100:.1f}%)"
    for tier, count in zip(sorted_tiers, sorted_counts)
]

ax.legend(wedges, legend_labels, title="Tiers", loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Player Tier Distribution – NA Solo/Duo (Pie Chart)", fontsize=14)
plt.tight_layout()
plt.savefig("plots/tier_pie_chart.png")
plt.close()