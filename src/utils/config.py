from pathlib import Path
from sumo_rl import parallel_env
from utils.sumo_multi_agent_env import SumoMultiAgentEnv

NETWORK = "jkpg"
NETWORK_DIR = Path("simulations/networks/jkpg")
RECORDINGS_DIR = Path("simulations/recordings")

DEFAULT_DELTA_TIME = 5
DEFAULT_NUM_SECONDS = 3600 + 4800 # 8400

def make_env(use_gui: bool = False, recording_name: str = None) -> SumoMultiAgentEnv:
    # Ensure recording directory exists
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    additional_cmd = ""
    if recording_name:
        additional_cmd = f"--tripinfo-output {RECORDINGS_DIR / recording_name}"

    raw_env = parallel_env(
        net_file=str(NETWORK_DIR / f"{NETWORK}.net.xml"),
        route_file=str(NETWORK_DIR / f"{NETWORK}.rou.xml"),
        use_gui=use_gui,
        sumo_warnings=True,
        delta_time=DEFAULT_DELTA_TIME,
        num_seconds=DEFAULT_NUM_SECONDS,
        additional_sumo_cmd=additional_cmd if additional_cmd else None
    )
    
    return SumoMultiAgentEnv(raw_env)