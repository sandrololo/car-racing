from environments.multiagent.cars import CarConfig, EnginePower, TyreType
from environments.multiagent.curriculum import CurriculumConfig, CurriculumStep

FIRST_TILE_VISITOR_REWARD_FACTOR = 1
OBS_FRAME_STACK = 4
NORMALIZE_REWARDS = True
NUM_ENV_RUNNERS = 4
HEAD_FCNET_HIDDENS = [128]
ROLLOUT_FRAGMENT_LENGTH = 600
TRAIN_MAX_TIMESTEPS_START = 500
TRAIN_MAX_TIMESTEPS_PER_EPISODE_INCREASE = 0.08
TRAIN_GAMMA = 0.98
TRAIN_NUM_ITERATIONS = 8000
MINI_BATCH_SIZE = 256
TRAIN_CLIP_PARAM = 0.1
TRAIN_NUM_EPOCHS = 5
LR_SCHEDULE_START = 0.00003
LR_SCHEDULE_END = 0.00003
EVAL_INTERVAL = 80
EVAL_MAX_TIMESTEPS = 800
EVAL_DURATION = 3
NUM_CARS = 1
CAR_CONFIGS = [
    CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
    CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
    CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
    CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
    CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
    CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
    CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
    CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
]
CURRICULUM_CONFIG = CurriculumConfig(
    CAR_CONFIGS,
    num_cars_start=1,
    entries=[
        CurriculumStep(num_cars=2, min_reward=600),
        CurriculumStep(num_cars=3, min_reward=1100),
        CurriculumStep(num_cars=4, min_reward=1600),
        CurriculumStep(num_cars=5, min_reward=2100),
        CurriculumStep(num_cars=6, min_reward=2600),
        CurriculumStep(num_cars=7, min_reward=3100),
        CurriculumStep(num_cars=8, min_reward=3600),
    ],
)
