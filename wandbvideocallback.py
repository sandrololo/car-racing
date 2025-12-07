import os
import glob
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms import Algorithm
import wandb

SINGLE_AGENT_VIDEO_DIR = "/tmp/single-agent-videos"
MULTI_AGENT_VIDEO_DIR = "/tmp/multi-agent-videos"
if not os.path.exists(SINGLE_AGENT_VIDEO_DIR):
    os.makedirs(SINGLE_AGENT_VIDEO_DIR)
for f in os.listdir(SINGLE_AGENT_VIDEO_DIR):
    os.remove(os.path.join(SINGLE_AGENT_VIDEO_DIR, f))
if not os.path.exists(MULTI_AGENT_VIDEO_DIR):
    os.makedirs(MULTI_AGENT_VIDEO_DIR)
for f in os.listdir(MULTI_AGENT_VIDEO_DIR):
    os.remove(os.path.join(MULTI_AGENT_VIDEO_DIR, f))


class _WandbVideoCallback(RLlibCallback):
    def __init__(self, video_dir: str, project_name: str):
        super().__init__()
        self.video_dir = video_dir
        self.project_name = project_name

    def on_evaluate_end(
        self, algorithm: Algorithm, metrics_logger, evaluation_metrics, **kwargs
    ):
        if not hasattr(self, "run"):
            api = wandb.Api()
            runs = api.runs(path=f"car-racing/{self.project_name}")
            self.run = wandb.init(
                id=runs[len(runs) - 1].id,
                project=self.project_name,
                resume="must",
            )
        if not os.path.isdir(self.video_dir):
            return
        files = []
        files.extend(glob.glob(os.path.join(self.video_dir, "*.mp4")))
        if not files:
            return
        filename = max(files, key=lambda p: os.path.getmtime(p))

        self.run.log(
            {"evaluation_video": wandb.Video(filename, format="mp4")},
            step=algorithm.iteration,
        )


class SingleAgentWandbVideoCallback(_WandbVideoCallback):
    def __init__(self):
        super().__init__(
            video_dir="/tmp/single-agent-videos",
            project_name="car-racing-single-agent",
        )


class MultiAgentWandbVideoCallback(_WandbVideoCallback):
    def __init__(self):
        super().__init__(
            video_dir="/tmp/multi-agent-videos",
            project_name="curriculum-car-racing-multi-agent",
        )
