from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.titlesize": 16})

def load_and_aggregate_runs(base_path_str: str, runs: int = 10) -> pd.DataFrame:
    """
    Loops through the 10 files for a model, merges them with a 'run' tracking column,
    and calculates calculated_avg_speed.
    """
    path_obj = Path(base_path_str)
    # Extracts everything before the extension to build the _X pattern
    dir_name = path_obj.parent
    base_name = path_obj.stem
    
    combined_dfs = []
    
    for r in range(1, runs + 1):
        run_file = dir_name / f"{base_name}_{r}.csv"
        if not run_file.exists():
            print(f"Warning: File missing: {run_file}")
            continue
            
        df = pd.read_csv(run_file, sep=";")
        df["run"] = r
        df["calculated_avg_speed"] = df["tripinfo_routeLength"] / df["tripinfo_duration"].replace(0, 1)
        combined_dfs.append(df)
        
    if not combined_dfs:
        raise FileNotFoundError(f"No run files found for base pattern: {base_path_str}")
        
    return pd.concat(combined_dfs, ignore_index=True)

def generate_comparison_dashboard(file_pattern_dict: dict, num_runs: int = 10) -> None:
    data_dict = {}
    for model_name, path_str in file_pattern_dict.items():
        print(f"Loading and processing 10 runs for {model_name}...")
        data_dict[model_name] = load_and_aggregate_runs(path_str, runs=num_runs)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Jönköping Network: Multi-Run Traffic Signal Comparison (10 Seeds)", weight="bold")
    colors = {"Baseline": "#7f8c8d", "DQN": "#e67e22", "PPO": "#2ecc71"}

    # number of arrivals
    ax1 = axes[0, 0]
    time_bins = np.arange(0, 8401, 60)
    
    for model_name, df in data_dict.items():
        binned_run_data = []
        unique_runs = df["run"].unique()
        
        for run_id in unique_runs:
            run_df = df[df["run"] == run_id].sort_values("tripinfo_arrival")
            counts = [np.searchsorted(run_df["tripinfo_arrival"].values, t) for t in time_bins]
            binned_run_data.append(counts)
            
        matrix = np.array(binned_run_data)
        mean_arrivals = np.mean(matrix, axis=0)
        std_arrivals = np.std(matrix, axis=0)
        
        confidence_interval = 1.96 * (std_arrivals / np.sqrt(len(unique_runs)))
        
        # Plot Mean Trend Line
        ax1.plot(time_bins, mean_arrivals, label=model_name, color=colors[model_name], linewidth=2.5, zorder=4)
        
        # FIX: Added an explicit translucent border to the shaded region so even tiny CIs are visible
        ax1.fill_between(
            time_bins, 
            mean_arrivals - confidence_interval, 
            mean_arrivals + confidence_interval, 
            color=colors[model_name], 
            alpha=0.20,
            edgecolor=colors[model_name],
            linewidth=0.5,
            linestyle="--"
        )
        
    ax1.set_title("Completed Trips Over Time (95% CI Shaded)", weight="bold")
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
            alpha=0.12,
            linewidth=2,
            ax=ax2
        )
    ax2.set_title("Global Time Loss Distribution", weight="bold")
    ax2.set_xlabel("Time Loss (Seconds)")
    ax2.set_ylabel("Density")
    ax2.set_xlim(0, 1000)
    ax2.legend()

    # standing time
    ax3 = axes[1, 0]
    all_stops = []
    all_waits = []
    
    for model_name, df in data_dict.items():
        run_means = df.groupby("run")[["tripinfo_waitingCount", "tripinfo_waitingTime"]].mean()
        
        global_stop_mean = run_means["tripinfo_waitingCount"].mean()
        global_wait_mean = run_means["tripinfo_waitingTime"].mean()
        
        stop_err = run_means["tripinfo_waitingCount"].std()
        wait_err = run_means["tripinfo_waitingTime"].std()
        
        # Handle nan or microscopic variance values safely for display
        stop_err = 0.0 if np.isnan(stop_err) else stop_err
        wait_err = 0.0 if np.isnan(wait_err) else wait_err
        
        all_stops.append(global_stop_mean)
        all_waits.append(global_wait_mean)
        
        ax3.errorbar(
            global_stop_mean, global_wait_mean,
            fmt='o', color=colors[model_name], ecolor='black', elinewidth=1.5, capsize=5,
            markersize=14, markeredgecolor='black', zorder=5, label=model_name
        )
        
        ax3.text(
            global_stop_mean + 0.2, 
            global_wait_mean + 1.0, 
            f"{model_name}\n({global_stop_mean:.2f} ± {stop_err:.2f} stops)\n({global_wait_mean:.1f} ± {wait_err:.1f}s)", 
            weight='bold' if model_name == "DQN" else 'normal',
            va='center',
            fontsize=10
        )
    ax3.margins(x=0.4, y=0.4)

    # vehicle velocity
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
    output_path = "simulations/recordings/multi_run_model_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Success! Final aggregated evaluation dashboard saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    base_files = {
        "Baseline": "simulations/recordings/jkpg_baseline_10s.csv", 
        "DQN":      "simulations/recordings/jkpg_dqn.csv",
        "PPO":      "simulations/recordings/jkpg_ppo.csv"
    }
    
    generate_comparison_dashboard(base_files, num_runs=10)