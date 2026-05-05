from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO

NETWORK = "network1"

env = SumoEnvironment(net_file=f"simulations/networks/{NETWORK}.net.xml",
                      route_file=f"simulations/networks/{NETWORK}.rou.xml", 
                      single_agent=True, 
                      use_gui=False, # not working on mac
                      sumo_warnings=True, 
                      additional_sumo_cmd=f"--tripinfo-output simulations/recordings/{NETWORK}.xml") # record data to analyze agent performance

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
model.save("ppo_agent")