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

from typing import Tuple

import gym
from gym import spaces
import numpy as np

from envs.factors.factor_wrapper import FactorWrapper


class LightWrapper(FactorWrapper):
  """Wrapper over MuJoCo environments that modifies lighting."""

  def __init__(self,
               env: gym.Env,
               diffuse_range: Tuple[float, float] = (0.2, 0.8),
               seed: int = None,
               **kwargs):
    """Creates a new wrapper."""
    super().__init__(
        env,
        factor_space=spaces.Box(low=np.array(diffuse_range[0]),
                                high=np.array(diffuse_range[1]),
                                dtype=np.float32,
                                seed=seed),
        **kwargs)

  @property
  def factor_name(self):
    return 'light'

  def _set_to_factor(self, value: float):
    """Sets to the given factor."""
    self.unwrapped.model.vis.headlight.ambient[:] = np.full((3, ), value)
    self.unwrapped.model.vis.headlight.diffuse[:] = np.full((3, ), value)

  def __getattr__(self, name: str):
    return getattr(self.env, name)
