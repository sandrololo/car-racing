import os
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms import Algorithm
import wandb
import glob


video_dir = os.path.join("/tmp", "videos")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
for f in os.listdir(video_dir):
    os.remove(os.path.join(video_dir, f))


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
        video_dir = os.path.join("/tmp", "videos")
        if not os.path.isdir(video_dir):
            return
        files = []
        files.extend(glob.glob(os.path.join(video_dir, "*.mp4")))
        if not files:
            return
        filename = max(files, key=lambda p: os.path.getmtime(p))

        self.run.log({"evaluation_video": wandb.Video(filename, format="mp4")})
