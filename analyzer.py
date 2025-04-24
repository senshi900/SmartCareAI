import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid

def analyze(df, output_dir):
    df["Center X"] = (df["X1"] + df["X2"]) / 2
    df["Center Y"] = (df["Y1"] + df["Y2"]) / 2
    df = df[df["Player ID"] != -1]

    plots = []

    # Health Analysis
    fatigue_data = []
    for player_id in df["Player ID"].unique():
        player_data = df[df["Player ID"] == player_id].sort_values("Frame")
        dx = player_data["Center X"].diff().fillna(0)
        dy = player_data["Center Y"].diff().fillna(0)
        movement = np.sqrt(dx**2 + dy**2)
        low_activity_frames = (movement < 1).sum()
        fatigue_data.append([player_id, movement.sum(), low_activity_frames])

    fatigue_df = pd.DataFrame(fatigue_data, columns=["Player ID", "Total Movement", "Low Activity Frames"])
    fatigue_df = fatigue_df.sort_values("Low Activity Frames", ascending=False)

    top_fatigued = fatigue_df.head(10)
    plt.figure(figsize=(10, 5))
    plt.bar(top_fatigued["Player ID"].astype(str), top_fatigued["Low Activity Frames"], color='red')
    plt.xlabel("Player ID")
    plt.ylabel("Low Activity Frames")
    plt.title("Top 10 Players with Fatigue Indicators")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fatigue_plot = os.path.join(output_dir, f"fatigue_{uuid.uuid4().hex[:6]}.png")
    plt.savefig(fatigue_plot)
    plots.append(fatigue_plot)
    plt.close()

    # Performance Analysis
    frame_counts = df["Player ID"].value_counts().reset_index()
    frame_counts.columns = ["Player ID", "Frames Appeared"]

    distances = {}
    for player_id in df["Player ID"].unique():
        player_data = df[df["Player ID"] == player_id].sort_values("Frame")
        dx = player_data["Center X"].diff().fillna(0)
        dy = player_data["Center Y"].diff().fillna(0)
        dist = np.sqrt(dx**2 + dy**2).sum()
        distances[player_id] = dist

    distance_df = pd.DataFrame(list(distances.items()), columns=["Player ID", "Distance Moved"])
    performance = pd.merge(frame_counts, distance_df, on="Player ID")

    top_distance = performance.sort_values("Distance Moved", ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    plt.bar(top_distance["Player ID"].astype(str), top_distance["Distance Moved"])
    plt.xlabel("Player ID")
    plt.ylabel("Distance Moved")
    plt.title("Top 10 Players by Distance Moved")
    plt.xticks(rotation=45)
    plt.tight_layout()
    dist_plot = os.path.join(output_dir, f"distance_{uuid.uuid4().hex[:6]}.png")
    plt.savefig(dist_plot)
    plots.append(dist_plot)
    plt.close()

    top_frames = performance.sort_values("Frames Appeared", ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    plt.bar(top_frames["Player ID"].astype(str), top_frames["Frames Appeared"], color='orange')
    plt.xlabel("Player ID")
    plt.ylabel("Frames Appeared")
    plt.title("Top 10 Players by Frame Presence")
    plt.xticks(rotation=45)
    plt.tight_layout()
    frame_plot = os.path.join(output_dir, f"frames_{uuid.uuid4().hex[:6]}.png")
    plt.savefig(frame_plot)
    plots.append(frame_plot)
    plt.close()

    # Injury Prediction
    injury_suspects = []
    for player_id in df["Player ID"].unique():
        player_data = df[df["Player ID"] == player_id].sort_values("Frame")
        dx = player_data["Center X"].diff().fillna(0)
        dy = player_data["Center Y"].diff().fillna(0)
        movement = np.sqrt(dx**2 + dy**2)
        total_movement = movement.sum()
        last_30 = movement[-30:]
        stopped = (last_30 < 1).sum()

        if total_movement > 100 and stopped >= 20:
            injury_suspects.append((player_id, round(total_movement, 2), stopped))

    injury_df = pd.DataFrame(injury_suspects, columns=["Player ID", "Total Movement", "Stoppage in Last 30 Frames"])
    if not injury_df.empty:
        player_id = injury_df.iloc[0]["Player ID"]
        stops = injury_df.iloc[0]["Stoppage in Last 30 Frames"]
        plt.figure(figsize=(6, 4))
        plt.bar([str(player_id)], [stops], color='indianred')
        plt.title("Potential Injury Detected")
        plt.xlabel("Player ID")
        plt.ylabel("Stoppage in Last 30 Frames")
        plt.text(0, stops + 1, f"{int(stops)} stops", ha='center', fontsize=12)
        plt.ylim(0, 30)
        plt.tight_layout()
        injury_plot = os.path.join(output_dir, f"injury_{uuid.uuid4().hex[:6]}.png")
        plt.savefig(injury_plot)
        plots.append(injury_plot)
        plt.close()

    # Match Statistics
    total_players = df["Player ID"].nunique()
    avg_frames = performance["Frames Appeared"].mean()
    avg_distance = performance["Distance Moved"].mean()
    max_player = performance.loc[performance["Distance Moved"].idxmax()]
    min_player = performance.loc[performance["Distance Moved"].idxmin()]

    labels = [
        "Total Players",
        "Avg Frames",
        "Avg Distance",
        f"Top Player (ID {int(max_player['Player ID'])})",
        f"Lowest Player (ID {int(min_player['Player ID'])})"
    ]
    values = [
        total_players,
        round(avg_frames, 2),
        round(avg_distance, 2),
        round(max_player["Distance Moved"], 2),
        round(min_player["Distance Moved"], 2)
    ]

    plt.figure(figsize=(10, 5))
    plt.barh(labels, values, color='teal')
    plt.xlabel("Value")
    plt.title("Match Statistics Summary")
    plt.tight_layout()
    match_plot = os.path.join(output_dir, f"match_stats_{uuid.uuid4().hex[:6]}.png")
    plt.savefig(match_plot)
    plots.append(match_plot)
    plt.close()

    return plots
