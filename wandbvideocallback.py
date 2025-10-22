from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms import Algorithm
import torch
import numpy as np
import wandb

from car_racing_env import CarRacingEnv, STATE_H, STATE_W


class WandbVideoCallback(RLlibCallback):

    def on_evaluate_end(
        self, algorithm: Algorithm, metrics_logger, evaluation_metrics, **kwargs
    ):
        if not hasattr(self, "run"):
            api = wandb.Api()
            runs = api.runs(path="car-racing/car-racing-single-agent")
            self.run = wandb.init(
                id=runs[len(runs) - 1].id,
                project="car-racing-single-agent",
                resume="must",
            )
        rl_model = algorithm.get_module()

        env = CarRacingEnv(
            config={
                "render_mode": None,
                "max_timesteps": 4000,
                "gray_scale": True,
                "frame_stack": 4,
                "frame_skip": 4,
            }
        )
        terminated, truncated = False, False
        total_reward = 0.0
        video_frames = []
        i = 0
        obs, info = env.reset()
        while not (terminated or truncated):
            obs_dict = {
                "obs": torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32)
            }
            result = rl_model.forward_inference(obs_dict)
            actions = result["action_dist_inputs"].numpy()[0]

            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward
            if i % 20 == 0:
                frame = obs[-1].reshape(
                    1, STATE_H, STATE_W
                )  # from 4x-stacked grayscale images to a single image
                frame = (frame * 255).astype(np.uint8)
                # convert to rgb
                frame = np.repeat(frame, 3, axis=0)
                video_frames.append(frame)
            i += 1

        self.run.log(
            {
                "evaluation_video": wandb.Video(
                    np.array(video_frames), fps=15, format="mp4"
                )
            }
        )
        wandb.log({"evaluation_video_reward": total_reward})
        env.close()
