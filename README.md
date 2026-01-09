# Multi Agent Car Racing Environment
<img width="1200" height="631" alt="Multi-agent CarRacing environment" src="https://github.com/user-attachments/assets/6306134e-f56c-47b2-aaf6-9e1dff2bf240" />

This repository contains a multi-agent version of the original [CarRacing environment](https://gymnasium.farama.org/environments/box2d/car_racing/). It is modified to support multiple agents while remaining as close as possible
to the original. This ensures that the existing dynamics and behavior of the environment are preserved. Consequently, the algorithms and techniques used to train the single-agent environment can be reused with minimal modifications.

## Using the environment
You can create the base environment directly and pass a config dict to control every built-in option. The environment accepts the following keys:
- `num_cars` (int): number of active agents (max 8).
- `car_configs` (list[CarConfig]): per-car setup; length should be at least `num_cars`.
- `lap_complete_percent` (float): percentage of total tiles visited in order for a lap to be considered complete.
- `first_tile_visitor_reward_factor` (float): scales the reward for the first agent that reaches a new tile.
- `render_mode` (str): one of `"human"`, `"state_pixels"`, `"video"`.

Example including the curriculum configuration used in training:

```python
from environments.multiagent import MultiAgentCarRacingEnv
from environments.multiagent.cars import CarConfig, EnginePower, TyreType
from environments.multiagent.curriculum import CurriculumConfig, CurriculumStep

car_configs = [
	CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
	CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
	CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
	CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
	CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
	CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
	CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
	CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
]

curriculum_config = CurriculumConfig(
	car_configs=car_configs,
	num_cars_start=1,
	entries=[
		CurriculumStep(num_cars=2, min_reward=650),
		CurriculumStep(num_cars=6, min_reward=1100),
	],
)

env = MultiAgentCarRacingEnv(
	config={
		"num_cars": 1,
		"car_configs": car_configs,
		"lap_complete_percent": 0.95,
		"first_tile_visitor_reward_factor": 1.0,
		"render_mode": "state_pixels",  # "human" for interactive window
		# Optional: curriculum, forwarded to callbacks if you use them
		"curriculum_config": curriculum_config,
	}
)
```

Common wrappers used in training (see `train-multi-ppo-curriculum.py`) can be applied on top: `GrayscaleObservation`, `FrameStackObservation`, `NormalizeObservation`, `NormalizeReward`, `IncreasingTimeLimit`, and `RecordVideo`.

## Results
### Single Agent

<img width="600" alt="image" src="https://github.com/user-attachments/assets/2e6b8adf-a06c-404e-aa2b-73c85a95b3e0" />

<img width="600" alt="Single Agent Training Result" src="https://github.com/user-attachments/assets/a3808901-e726-466c-a85f-991f1c324c09" />

Training result using RLlib with the PPO algorithm and the original CarRacing environment. Another video can be found [here](https://github.com/user-attachments/assets/8f3fdaf5-1c18-4038-bb15-1740ffcb8b06) 

### Multi Agent

<img width="600" alt="Multi Agent Training Result" src="https://github.com/user-attachments/assets/6bfd2646-d3e6-4af6-aa8d-ff6a1b242fb6" />

Training result with curriculum learning using RLlib with the PPO algorithm in the specifig multi-agent environment.



## Installation
### Swig
Swig is required to install the *box2d* feature for gymnasium environments.
Swig is the abbreviation of *Simplified Wrapper and Interface Generator*, it can give script language such as python the ability to invoke C and C++ libraries interface method indirectly.

[](https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/) (On Windows make sure to download **swig**win-xxx which includes a prebuild executable)

### Python Environment
The recommended approach is to use Pipenv to create a virtual environment and install the dependencies.
Otherwise, the necessary packages are defined in `Pipfile`.

```bash
pip install pipenv
pipenv install
pipenv shell
```

## Training
It is recommended to run the training on a machine with 8 cores and one available GPU. Weights-and-Biases logging
is turned on per default, therefore being logged into wandb is a prerequisite.

### Single Agent
`python train-single-ppo.py`

### Multi Agent
`python train-multi-ppo.py`

### Multi Agent with Curriculum
`python train-multi-ppo-curriculum.py`
