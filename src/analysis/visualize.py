from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.titlesize": 16})

def generate_comparison_dashboard(file_dict: dict) -> None:
    """
    file_dict: A dictionary mapping model names to their CSV paths.
    """
    data_dict = {}
    for model_name, path_str in file_dict.items():
        path = Path(path_str)
        if not path.exists():
            print(f"{model_name} file not found at {path_str}")
            return
        
        df = pd.read_csv(path, sep=";")
        df["calculated_avg_speed"] = df["tripinfo_routeLength"] / df["tripinfo_duration"].replace(0, 1)
        df = df.sort_values(by="tripinfo_arrival").reset_index(drop=True)
        df["cumulative_arrivals"] = range(1, len(df) + 1)
        data_dict[model_name] = df

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Jönköping Network: Traffic Signal Algorithm Comparison", weight="bold")
    colors = {"Baseline": "#7f8c8d", "DQN": "#e67e22", "PPO": "#2ecc71"}

    # cumulative arrivals
    ax1 = axes[0, 0]
    for model_name, df in data_dict.items():
        ax1.plot(
            df["tripinfo_arrival"],
            df["cumulative_arrivals"],
            label=model_name,
            color=colors[model_name],
            linewidth=2.5
        )
    ax1.set_title("Completed Trips Over Time", weight="bold")
    ax1.set_xlabel("Simulation Time (Seconds)")
    ax1.set_ylabel("Accumulated Completed Trips")
    ax1.set_xlim(0, 8400)
    ax1.legend()

    # time loss
    ax2 = axes[0, 1]
    for model_name, df in data_dict.items():
        sns.kdeplot(
            data=df,
            x="tripinfo_timeLoss",
            label=model_name,
            color=colors[model_name],
            fill=True,
            alpha=0.15,
            linewidth=2,
            ax=ax2
        )
    ax2.set_title("Time Loss Distribution", weight="bold")
    ax2.set_xlabel("Time Loss (Seconds)")
    ax2.set_ylabel("Density")
    ax2.set_xlim(0, 1000)
    ax2.legend()

    # waiting time vs number of stops
    ax3 = axes[1, 0]
    for model_name, df in data_dict.items():
        avg_stops = df["tripinfo_waitingCount"].mean()
        avg_wait = df["tripinfo_waitingTime"].mean()
        ax3.scatter(
            avg_stops, 
            avg_wait, 
            label=model_name, 
            color=colors[model_name], 
            s=250,
            edgecolor='black', 
            zorder=5
        )
        ax3.text(
            avg_stops + 0.05, 
            avg_wait + 1.0, 
            f" {model_name}\n ({avg_stops:.2f} stops, {avg_wait:.1f}s)", 
            weight='bold' if model_name == "PPO" else 'normal',
            va='center'
        )
    ax3.set_title("Standing Time Across Stops", weight="bold")
    ax3.set_xlabel("Average Number of Stops per Vehicle")
    ax3.set_ylabel("Average Standing Time per Vehicle (Seconds)")
    # add padding on axes so text labels don't get cut off
    all_stops = [df["tripinfo_waitingCount"].mean() for df in data_dict.values()]
    all_waits = [df["tripinfo_waitingTime"].mean() for df in data_dict.values()]
    ax3.set_xlim(min(all_stops) - 0.5, max(all_stops) + 1.0)
    ax3.set_ylim(min(all_waits) - 10, max(all_waits) + 20)
    ax3.grid(True, linestyle="--", alpha=0.7)

    # average speed
    ax4 = axes[1, 1]
    for model_name, df in data_dict.items():
        sns.regplot(
            data=df,
            x="tripinfo_routeLength",
            y="calculated_avg_speed",
            scatter=False,
            label=model_name,
            color=colors[model_name],
            ax=ax4,
            line_kws={"linewidth": 2.5}
        )
    ax4.set_title("Velocity Across Route Distance", weight="bold")
    ax4.set_xlabel("Trip Route Length (Meters)")
    ax4.set_ylabel("Average Trip Speed (m/s)")
    ax4.legend()

    plt.tight_layout()
    output_path = "simulations/recordings/model_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    files = {
        "Baseline": "simulations/recordings/jkpg_baseline_10s.csv",
        "DQN":      "simulations/recordings/jkpg_dqn.csv",
        "PPO":      "simulations/recordings/jkpg_ppo.csv"
    }
    
    generate_comparison_dashboard(files)