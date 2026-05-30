from pathlib import Path
from sumo_rl import parallel_env
from envs.gridlock_monitor import GridlockMonitor

NETWORK = "jkpg"
NETWORK_DIR = Path("simulations/networks/jkpg")
RECORDINGS_DIR = Path("simulations/recordings")

DEFAULT_DELTA_TIME = 5
DEFAULT_NUM_SECONDS = 8400

def make_env(scale: float = 1, max_halting_vehicles: int = 800, use_gui: bool = False, recording_name: str = None):
    # Ensure recording directory exists
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    additional_cmd = f"--scale {scale}"
    if recording_name:
        additional_cmd += f" --tripinfo-output {RECORDINGS_DIR / recording_name}"

    raw_env = parallel_env(
        net_file=str(NETWORK_DIR / f"{NETWORK}.net.xml"),
        route_file=str(NETWORK_DIR / f"{NETWORK}.rou.xml"),
        use_gui=use_gui,
        sumo_warnings=True,
        delta_time=DEFAULT_DELTA_TIME,
        num_seconds=DEFAULT_NUM_SECONDS,
        additional_sumo_cmd=additional_cmd if additional_cmd else None,
        reward_fn = "queue"
    )
    
    return GridlockMonitor(raw_env, max_halting_vehicles=max_halting_vehicles)