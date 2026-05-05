from sumo_rl import SumoEnvironment

env = SumoEnvironment(net_file="simulations/networks/network1.net.xml",
                      route_file="simulations/test.rou.xml", 
                      single_agent=True, 
                      use_gui=False, # not working on mac
                      sumo_warnings=True)

# Reset the environment to start
obs, info = env.reset()
print("observation space:", env.observation_space)
print("action space:", env.action_space)

# Run for 10 steps to see cars move
for i in range(10):
    # Sample a random action (e.g., change light or stay)
    action = env.action_space.sample()
    
    # Apply the action
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {i}: Reward received: {reward}")

env.close()