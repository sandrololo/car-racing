from environments.multiagent.cars import CarConfig, EnginePower, TyreType
from environments.multiagent.curriculum import CurriculumConfig, CurriculumStep

FIRST_TILE_VISITOR_REWARD_FACTOR = 1
OBS_FRAME_STACK = 4
NORMALIZE_REWARDS = True
NUM_ENV_RUNNERS = 4
HEAD_FCNET_HIDDENS = [128]
ROLLOUT_FRAGMENT_LENGTH = 800
TRAIN_MAX_TIMESTEPS_START = 500
TRAIN_MAX_TIMESTEPS_PER_EPISODE_INCREASE = 0.1
TRAIN_GAMMA = 0.98
TRAIN_NUM_ITERATIONS = 2000
MINI_BATCH_SIZE = 256
TRAIN_CLIP_PARAM = 0.1
TRAIN_NUM_EPOCHS = 5
LR_SCHEDULE_START = 0.0001
LR_SCHEDULE_END = 0.0001
EVAL_INTERVAL = 50
EVAL_MAX_TIMESTEPS = 800
EVAL_DURATION = 3
NUM_CARS = 4
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
        CurriculumStep(num_cars=3, min_reward=1200),
        CurriculumStep(num_cars=4, min_reward=1800),
    ],
)
