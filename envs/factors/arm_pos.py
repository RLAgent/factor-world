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


class ArmPosWrapper(FactorWrapper):
  """Wrapper over MuJoCo environments that modifies arm position."""

  def __init__(self,
               env: gym.Env,
               x_range: Tuple[float, float] = (-0.5, 0.5),
               y_range: Tuple[float, float] = (-0.2, 0.4),
               z_range: Tuple[float, float] = (-0.15, 0.1),
               seed: int = None,
               **kwargs):
    """Creates a new wrapper."""
    # Note: Must set this to None before calling super()__init__().
    self._default_hand_init_pos = None
    super().__init__(
        env,
        factor_space=spaces.Box(
            low=np.array([x_range[0], y_range[0], z_range[0]]),
            high=np.array([x_range[1], y_range[1], z_range[1]]),
            dtype=np.float32,
            seed=seed),
        **kwargs)

    # Store default values
    self._default_hand_init_pos = self.unwrapped.hand_init_pos.copy()

  @property
  def factor_name(self):
    return 'arm_pos'

  def _set_to_factor(self, delta_pos: Tuple[float, float, float]):
    """Sets to the given factor."""
    if self._default_hand_init_pos is None:
      print("Warning: Default positions not set. Not setting factor value.")
      return

    self.unwrapped.hand_init_pos[:] = self._default_hand_init_pos + delta_pos
    self.unwrapped._reset_hand()
    self.unwrapped.init_tcp = self.unwrapped.tcp_center
    self.unwrapped.init_left_pad = self.unwrapped.get_body_com('leftpad')
    self.unwrapped.init_right_pad = self.unwrapped.get_body_com('rightpad')
