import os
import glob
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms import Algorithm
import wandb

SINGLE_AGENT_VIDEO_DIR = "/tmp/single-agent-videos"
MULTI_AGENT_VIDEO_DIR = "/tmp/multi-agent-videos"
CURRICULUM_MULTI_AGENT_VIDEO_DIR = "/tmp/multi-agent-videos-curriculum"


class _WandbVideoCallback(RLlibCallback):
    def __init__(self, video_dir: str, project_name: str):
        super().__init__()
        self.video_dir = video_dir
        self.project_name = project_name
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        for f in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, f))

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
            video_dir=SINGLE_AGENT_VIDEO_DIR,
            project_name="car-racing-single-agent",
        )


class MultiAgentWandbVideoCallback(_WandbVideoCallback):
    def __init__(self):
        super().__init__(
            video_dir=MULTI_AGENT_VIDEO_DIR,
            project_name="car-racing-multi-agent",
        )


class MultiAgentCurriculumWandbVideoCallback(_WandbVideoCallback):
    def __init__(self):
        super().__init__(
            video_dir=CURRICULUM_MULTI_AGENT_VIDEO_DIR,
            project_name="curriculum-car-racing-multi-agent",
        )


class NetworkSummaryCallback(RLlibCallback):
    def on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        **kwargs,
    ) -> None:
        algorithm.learner_group.foreach_learner(lambda l: print(l.module.__repr__()))
