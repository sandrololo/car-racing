import logging
import time
import numpy as np
import wandb
import torch
from ray.rllib.algorithms.ppo import PPO
from environments.singleagent.environment import SingleAgentCarRacingEnv
import single_agent_config

CHECKPOINT_NAME = "checkpoint_PPO_SingleAgentCarRacingEnv_febbe_00000"

run = wandb.init()
artifact = run.use_artifact(
    f"car-racing/car-racing-single-agent/{CHECKPOINT_NAME}:v0", type="model"
)
artifact_dir = artifact.download()

logger = logging.getLogger(__name__)


def render_model():
    print("Loading trained model...")

    algo = PPO.from_checkpoint(artifact_dir)
    print("Model loaded successfully!")

    env = SingleAgentCarRacingEnv(
        config={
            "render_mode": "human",
            "gray_scale": single_agent_config.OBS_GRAY_SCALE,
            "frame_stack": single_agent_config.OBS_FRAME_STACK,
        }
    )
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

        time.sleep(0.005)  # Slower for visibility

    print(f"\nCompleted! Total reward: {total_reward:.2f}")
    env.close()
    algo.stop()


if __name__ == "__main__":
    render_model()
