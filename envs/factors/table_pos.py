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

from copy import copy
from typing import Tuple

import gym
from gym import spaces
import numpy as np

from envs.factors import constants
from envs.factors.factor_wrapper import FactorWrapper


class TablePosWrapper(FactorWrapper):
  """Wrapper over MuJoCo environments that modifies table position."""

  def __init__(self,
               env: gym.Env,
               x_range: Tuple[float, float] = (-0.025, 0.025),
               y_range: Tuple[float, float] = (-0.1, 0.1),
               z_range: Tuple[float, float] = (-0.08, 0),
               seed: int = None,
               **kwargs):
    """Creates a new wrapper."""
    # Note: Must set this to None before calling super()__init__().
    self._default = None
    super().__init__(
        env,
        factor_space=spaces.Box(
            low=np.array([x_range[0], y_range[0], z_range[0]]),
            high=np.array([x_range[1], y_range[1], z_range[1]]),
            dtype=np.float32,
            seed=seed),
        **kwargs)

    # Store default values
    self._default = {}
    self._default['obj_init_pos'] = self.unwrapped.obj_init_pos.copy()
    if hasattr(self.unwrapped, 'goal'):
      self._default['goal'] = self.unwrapped.goal.copy()
    if hasattr(self.unwrapped, 'goal_space'):
      self._default['goal_space'] = copy(self.unwrapped.goal_space)
    self._default['_random_reset_space'] = copy(
        self.unwrapped._random_reset_space)
    self._default['_HAND_SPACE'] = copy(self.unwrapped._HAND_SPACE)
    self._default['hand_low'] = self.unwrapped.hand_low.copy()
    self._default['hand_high'] = self.unwrapped.hand_high.copy()
    self._default['hand_init_pos'] = self.unwrapped.hand_init_pos.copy()
    self._default['mocap_low'] = self.unwrapped.mocap_low.copy()
    self._default['mocap_high'] = self.unwrapped.mocap_high.copy()

    for body_name in constants.TABLE_BODY_NAMES:
      if body_name in self.model.body_names:
        id = self.model.body_name2id(body_name)
        self._default[body_name] = self.model.body_pos[id].copy()

  @property
  def factor_name(self):
    return 'table_pos'

  def _set_to_factor(self, delta_pos: Tuple[float, float, float]):
    """Sets to the given factor."""
    if not self._default:
      print("WARNING(table_pos): Default positions not set. "
            "Not setting factor value.")
      return

    # Change table position.
    for body_name in constants.TABLE_BODY_NAMES:
      if body_name in self.model.body_names:
        id = self.model.body_name2id(body_name)
        self.unwrapped.model.body_pos[id] = self._default[body_name] + delta_pos

    # Change init position of the object.
    obj_init_pos = self._default['obj_init_pos'] + delta_pos
    self.unwrapped.init_config['obj_init_pos'] = obj_init_pos
    self.unwrapped.obj_init_pos = obj_init_pos

    if hasattr(self.unwrapped, 'distractor_init_pos'):
      self.unwrapped.distractor_init_pos += delta_pos

    # Change init position of the hand.
    self.unwrapped.hand_init_pos = self._default['hand_init_pos'] + delta_pos

    # Change goal position.
    if hasattr(self.unwrapped, 'goal'):
      goal_pos = self._default['goal'] + delta_pos
      self.unwrapped.goal = goal_pos

    # Change reset space.
    assert len(self._default['_random_reset_space'].shape) == 1
    assert self._default['_random_reset_space'].shape[0] % 3 == 0
    num_xyz = self._default['_random_reset_space'].shape[0] // 3
    self.unwrapped._random_reset_space = spaces.Box(
        self._default['_random_reset_space'].low + np.tile(delta_pos, num_xyz),
        self._default['_random_reset_space'].high +
        np.tile(delta_pos, num_xyz),
        self._default['_random_reset_space'].shape,
        self._default['_random_reset_space'].dtype)
    if hasattr(self.unwrapped, 'goal_space'):
      self.unwrapped.goal_space = spaces.Box(
          self._default['goal_space'].low + delta_pos,
          self._default['goal_space'].high + delta_pos,
          self._default['goal_space'].shape,
          self._default['goal_space'].dtype)

    # Change range of mocap.
    self.unwrapped.hand_low = self._default['hand_low'] + delta_pos
    self.unwrapped.hand_high = self._default['hand_high'] + delta_pos
    self.unwrapped.mocap_low = self._default['mocap_low'] + delta_pos
    self.unwrapped.mocap_high = self._default['mocap_high'] + delta_pos
    self.unwrapped._HAND_SPACE = spaces.Box(
        self._default['_HAND_SPACE'].low + delta_pos,
        self._default['_HAND_SPACE'].high + delta_pos,
        self._default['_HAND_SPACE'].shape,
        self._default['_HAND_SPACE'].dtype)

    self.reset_model()
