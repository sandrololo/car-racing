import logging
import os
import glob
import time
import numpy as np
import torch
from car_racing_env import CarRacingEnv
from ray.rllib.algorithms.ppo import PPO

logger = logging.getLogger(__name__)

def find_latest_checkpoint():
    results_dir = os.path.abspath("results/single-agent")
    ppo_dirs = glob.glob(os.path.join(results_dir, "PPO_*"))
    ppo_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = ppo_dirs[0]
    
    trial_dirs = glob.glob(os.path.join(latest_dir, "PPO_CarRacingEnv_*"))
    trial_dir = trial_dirs[0]
    
    checkpoint_dirs = glob.glob(os.path.join(trial_dir, "checkpoint_*"))
    checkpoint_dirs.sort(key=lambda x: int(x.split('_')[-1]))
    return os.path.abspath(checkpoint_dirs[-1])


def render_model():
    print("Loading trained model...")
    
    checkpoint_path = find_latest_checkpoint()
    print(f"Loading from: {checkpoint_path}")
    
    algo = PPO.from_checkpoint(checkpoint_path)
    print("Model loaded successfully!")
    
    env = CarRacingEnv(config={"render_mode": "human"})
    obs, info = env.reset()

    total_reward = 0
    terminated, truncated = False, False
    
    while not (terminated or truncated):
        rl_module = algo.get_module()
        obs_dict = {"obs": torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32)}
        result = rl_module.forward_inference(obs_dict)
        actions = result["action_dist_inputs"].numpy()[0]
        
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        env.render()    
        
        time.sleep(0.01)  # Slower for visibility
    
    print(f"\nCompleted! Total reward: {total_reward:.2f}")
    env.close()
    algo.stop()


if __name__ == "__main__":
    render_model()
