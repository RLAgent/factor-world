# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Tuple

import gym
from gym import spaces
from gym.wrappers import ClipAction, TimeLimit
import numpy as np

from envs.env_dict import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from envs.factors.utils import make_env_with_factors


class MetaWorldState(gym.ObservationWrapper):
  def __init__(self, env: gym.Env, state_key: str = "proprio"):
    super().__init__(env)
    self.state_key = state_key

    state_space = spaces.Box(
        low=self.state(self.observation_space[self.state_key].low),
        high=self.state(self.observation_space[self.state_key].high))

    self.observation_space = spaces.Dict({
        key: state_space if key == self.state_key else space
        for key, space in self.observation_space.spaces.items()
    })

  @staticmethod
  def state(state):
    return np.concatenate((state[..., :4], state[..., 18:22]), axis=-1)

  def observation(self, observation):
    """Removes object information from the state.
    Args:
        observation: The observation to remove object information from
    Returns:
        The updated observation
    """
    observation.update({
        self.state_key: self.state(observation[self.state_key]),
    })
    return observation


def make_wrapped_env(
    env_name: str,
    use_train_xml: bool,
    factor_kwargs: Dict[str, Dict[str, Any]] = {},
    image_obs_size: Tuple[int, int] = (84, 84),
    camera_name: str = 'corner3',
    observe_goal: bool = False,
    remove_object_from_state: bool = False,
    clip_action: bool = True,
    default_num_resets_per_randomize: int = 1,
) -> gym.Env:
  """Create a wrapped environment.

  Args:
    env_name: Environment name
    factor_kwargs: Dictionary of factor_name -> kwargs
    use_train_xml: Whether to load train or eval XML (e.g., textures).
    image_obs_size: Image obs resolution. Set to None to exclude
        images from observations.
    camera_name: Name of camera viewpoint
    observe_goal: Whether to include goal in the observation
    remove_object_from_state: Whether to remove object info from state
    clip_action: Whether to clip actions
    num_resets_per_randomize: Default number of env resets before
        randomizing a factor.
  """
  if observe_goal:
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name +
                                                  '-goal-observable']
  else:
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_name + '-goal-hidden']

  env_kwargs = dict(
      camera_name=camera_name,
      get_image_obs=(image_obs_size is not None),
      image_obs_size=image_obs_size,
  )

  factor_kwargs = factor_kwargs.copy()
  for i, factor in enumerate(factor_kwargs.keys()):
    if factor_kwargs[factor]['num_resets_per_randomize'] == 'default':
      factor_kwargs[factor]['num_resets_per_randomize'] = default_num_resets_per_randomize

  # Wrap env with factors of variation.
  env = make_env_with_factors(
      env_cls, env_kwargs,
      factor_kwargs=factor_kwargs,
      use_train_xml=use_train_xml)

  if remove_object_from_state:
    env = MetaWorldState(env)

  if clip_action:
    env = ClipAction(env)

  max_episode_steps = env.max_path_length
  env = TimeLimit(env, max_episode_steps=max_episode_steps)

  return env
