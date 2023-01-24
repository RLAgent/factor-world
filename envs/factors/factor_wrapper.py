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

"""Environment wrappers for factors of variation.

Taken from https://github.com/google-research/weakly_supervised_control/
  File: weakly_supervised_control/envs/env_wrapper.py
"""

import abc
from typing import Any, Dict, List, Sequence, Union

import gym
from gym import spaces
import numpy as np


def sample_without_replacement(factor_space: spaces.Space, n: int):
  """Samples from factor_space without replacement.

  If factor_space is spaces.Dict, then returns a dict of
  str -> sampled values.
  """
  if isinstance(factor_space, spaces.Box):
    return np.stack([
        np.atleast_1d(factor_space.sample())
        for i in range(n)
    ])
  elif isinstance(factor_space, spaces.Discrete):
    values = np.arange(
        factor_space.start,
        factor_space.start + factor_space.n)
    return np.random.choice(values, size=n, replace=False)
  elif isinstance(factor_space, spaces.Dict):
    return {
        key: sample_without_replacement(factor_space[key], n)
        for key in factor_space
    }


def dict_to_list_of_dicts(
        d: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
  """Converts a dictionary of str -> values to a list of dicts of str -> value.

  Example:
    Input: {
      'a': [[0, 1], [2, 3]],
      'b': [[4, 5], [6, 7]],
    }
    Output: [
      {'a': [0, 1], 'b': [4, 5]},
      {'a': [2, 3], 'b': [6, 7]},
    ]

  NOTE: d[key] must be of same length for all keys.
  """
  lengths = [len(d[key]) for key in d]
  assert len(set(lengths)) == 1  # All lengths should be equal.
  return [
      {
          key: d[key][i] for key in d
      } for i in range(lengths[0])
  ]


class FactorWrapper:
  """Wrapper to add randomized factors to an environment."""

  def __init__(self,
               env: gym.Env,
               factor_space: spaces.Box,
               init_factor_value: np.ndarray = None,
               num_resets_per_randomize: int = 1):
    """Adds randomized factors to an environment.

    Args:
      env: The gym environment to wrap.
      factor_space: Space to sample factor values from.
      init_factor_value: Initial factor value. If None, samples a random
          factor value.
      num_resets_per_randomize: Number of reset() calls until a new factor
          value is sampled. If None, never randomizes the factor.
    """
    super().__init__()
    self.env = env
    self.factor_space = factor_space
    self._current_factor_value = None

    if init_factor_value is not None:
      self._set_and_cache_factor(init_factor_value)
    else:
      self.randomize_factor()

    self._num_resets_per_randomize = num_resets_per_randomize
    self._num_resets = 0

  @abc.abstractproperty
  def factor_name(self) -> str:
    """String name of the factor."""

  @abc.abstractmethod
  def sample_factor_value(
          self, num_values: int) -> Union[np.ndarray, Dict[str, Any]]:
    """Returns N sampled factor values over the factor space."""

  @property
  def factor_names(self) -> Sequence[str]:
    """Recursively gets factor_names of all unwrapped envs"""
    factor_names = []
    if isinstance(self.env, FactorWrapper):
      factor_names = self.env.factor_names
    return factor_names + [self.factor_name]

  @property
  def current_factor_value(self):
    """Current factor value."""
    return self._current_factor_value

  @property
  def current_factor_values(self) -> Dict[str, np.ndarray]:
    """Recursively gets the current factor values of all unwrapped envs"""
    factor_values = {self.factor_name: self.current_factor_value}
    if isinstance(self.env, FactorWrapper):
      factor_values.update(self.env.current_factor_values)
    return factor_values

  def sample_factor_value(self, n: int) -> np.ndarray:
    """Returns N sampled factor values over the factor space.

    Samples discrete factors without replacement.

    Args:
      n: Number of factor values
    Returns:
      A list of factor values.
    """
    factor_values = sample_without_replacement(self.factor_space, n)
    if isinstance(factor_values, dict):
      # Convert to a list of dicts.
      return dict_to_list_of_dicts(factor_values)
    else:
      return factor_values

  @abc.abstractmethod
  def _set_to_factor(self, value: Union[np.ndarray, Any]):
    """Sets to the given factor."""

  def set_factor_values(self, values: Dict[str, np.ndarray]):
    """Sets to the given factor values.

    Args:
      values: Dictionary of factor name -> factor values
    """
    if isinstance(self.env, FactorWrapper):
      self.env.set_factor_values(values)
    self._set_and_cache_factor(values[self.factor_name])

  def randomize_factor(self):
    # TODO(lslee): For discrete factors, make sure not to sample a
    # previously-seen value.
    factor = self.sample_factor_value(1)[0]
    self._set_and_cache_factor(factor)

  def _set_and_cache_factor(self, value: Union[np.ndarray, Dict[str, Any]]):
    self._set_to_factor(value)
    self._current_factor_value = value

  # ----------------------------------------------------------------
  # Gym Env
  # ----------------------------------------------------------------

  def step(self, action):
    return self.env.step(action)

  def reset(self, force_randomize_factor: bool = False):
    # NOTE: We reset the base (unwrapped) env first, in case the
    #       XML neeeds to be reloaded.
    reset_kwargs = {}
    if isinstance(self.env, FactorWrapper):
      reset_kwargs = dict(force_randomize_factor=force_randomize_factor)
    obs = self.env.reset(**reset_kwargs)

    # Randomize factor.
    if force_randomize_factor or (
            self._num_resets_per_randomize and
            self._num_resets % self._num_resets_per_randomize == 0):
      self.randomize_factor()

    self._num_resets += 1
    return self.env._get_obs()

  @property
  def action_space(self):
    return self.env.action_space

  @property
  def observation_space(self):
    return self.env.observation_space

  def render(self, **kwargs):
    return self.env.render(**kwargs)

  # ----------------------------------------------------------------
  # SawyerXYZEnv
  # ----------------------------------------------------------------

  def close(self, **kwargs):
    return self.env.close(**kwargs)

  def set_state(self, qpos, qvel):
    return self.env.set_state(qpos, qvel)

  @property
  def unwrapped(self):
    return self.env.unwrapped

  @property
  def model(self):
    return self.env.model

  def _get_obs(self):
    return self.env._get_obs()

  @property
  def spec(self):
    return self.env.spec

  def reset_model(self, **kwargs):
    return self.env.reset_model(**kwargs)

  @property
  def max_path_length(self):
    return self.env.max_path_length

  @property
  def data(self):
    return self.env.data

  @property
  def metadata(self):
    return self.env.metadata

  # ----------------------------------------------------------------
  # XmlWrapper
  # ----------------------------------------------------------------

  @property
  def distractor_name(self):
    return self.env.distractor_name

  @property
  def object_name(self):
    return self.env.object_name


if __name__ == '__main__':
  """
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so:/usr/lib/x86_64-linux-gnu/libGLEW.so \
  python -m envs.factors.factor_wrapper
  """
  from envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
  from envs.factors.light import LightWrapper

  env_name = 'pick-place-v2'
  camera_name = 'corner3'

  env_constructor = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_name + '-goal-hidden']
  base_env = env_constructor(camera_name=camera_name,
                             get_image_obs=False)

  env = LightWrapper(base_env)

  for e in range(10):
    obs = env.reset()

    for _ in range(100):
      env.step(env.action_space.sample())
      env.render()
