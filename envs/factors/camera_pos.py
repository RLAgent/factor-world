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


class CameraPosWrapper(FactorWrapper):
  """Wrapper over MuJoCo environments that modifies camera position."""

  def __init__(self,
               env: gym.Env,
               azimuth_range: Tuple[float, float] = (np.pi / 2, 3 * np.pi / 4),
               inclination_range: Tuple[float, float] = (np.pi / 6, np.pi / 3),
               radius_range: Tuple[float, float] = (1.25, 1.75),
               seed: int = None,
               **kwargs):
    """Creates a new wrapper."""
    super().__init__(
        env,
        factor_space=spaces.Box(
            low=np.array([
                azimuth_range[0], inclination_range[0], radius_range[0]]),
            high=np.array([
                azimuth_range[1], inclination_range[1], radius_range[1]]),
            dtype=np.float32,
            seed=seed),
        **kwargs)

  @property
  def factor_name(self):
    return 'camera_pos'

  def _set_to_factor(self, value: Tuple[float, float, float]):
    """Sets to the given factor."""
    assert self.unwrapped._camera_name == "movable"
    cam_id = self.unwrapped.model.camera_name2id(self.unwrapped._camera_name)

    phi, theta, rad = value[0], value[1], value[2]
    pos = np.array([
        rad * np.cos(phi) * np.sin(theta),
        rad * np.sin(phi) * np.sin(theta),
        rad * np.cos(theta),
    ])
    self.unwrapped.model.cam_pos[cam_id][:] = pos
