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
from scipy.spatial.transform import Rotation

from envs.factors import constants
from envs.factors.env_utils import get_table_pos
from envs.factors.factor_wrapper import FactorWrapper


class DistractorPosWrapper(FactorWrapper):
  """Wrapper over MuJoCo environments that modifies distractor object position."""

  def __init__(self,
               env: gym.Env,
               table_edge_distance: float = 0.05,
               theta_range: Tuple[float, float] = (0, 2 * np.pi),
               seed: int = None,
               **kwargs):
    """Creates a new wrapper.

    Args:
      env: The environment. Must be wrapped with
          envs.factors.distractor_wrapper.DistractorWrapper
      edge_distance: Min distance from edges of the table
      seed: Random seed
    """
    self._seed = seed
    self.edge_distance = table_edge_distance

    # Get XY range based on table size.
    table_size = constants.DEFAULT_TABLE_SIZE[:2]
    xy_low = -table_size / 2 + table_edge_distance
    xy_high = +table_size / 2 - table_edge_distance

    # Range for XY (position) and radian (rotation)
    xyr_low = np.append(
        xy_low, [theta_range[0]])
    xyr_high = np.append(
        xy_high, [theta_range[1]])

    self.distractor_init_quat = [1, 0, 0, 0]
    super().__init__(
        env,
        factor_space=spaces.Box(
            low=xyr_low,
            high=xyr_high,
            dtype=np.float32,
            seed=seed),
        **kwargs)

    self.i_qp = self.model.get_joint_qpos_addr(
        f"joint_{self.distractor_name}")[0]
    self.i_qv = self.model.get_joint_qvel_addr(
        f"joint_{self.distractor_name}")[0]

    # # Store init pos & quat of distractor object
    qpos = self.unwrapped.data.qpos.flat.copy()
    self.distractor_init_pos = qpos[self.i_qp:self.i_qp + 3]  # Unused
    self.distractor_init_quat = qpos[self.i_qp + 3:self.i_qp + 7]

  @property
  def factor_name(self):
    return 'distractor_pos'

  def _set_to_factor(self, value: Tuple[float, float]):
    """Sets to the given factor.

    Args:
      value: (N, 3) matrix containing (X, Y, theta)
             of each distractor object relative to
             the center of the table.
    """
    if (not hasattr(self, 'i_qp') or
            not hasattr(self, 'i_qv')):
      print("WARNING(distractor_pos): Missing i_qp/i_qv. Not setting factor.")
      return

    delta_xy = value[:2]
    delta_radians = value[2]

    table_pos = get_table_pos(self.env)

    delta_pos = np.concatenate([delta_xy, [0]])
    pos = table_pos + delta_pos

    radians = Rotation.from_quat(self.distractor_init_quat).as_euler('xyz')
    radians[0] += delta_radians
    quat = Rotation.from_euler('xyz', radians).as_quat()

    # Set distractor object position.
    qpos = self.data.qpos.flat.copy()
    qvel = self.data.qvel.flat.copy()
    qpos[self.i_qp:self.i_qp + 3] = pos
    qpos[self.i_qp + 3:self.i_qp + 7] = quat
    qvel[self.i_qv:self.i_qv + 3] = [0, 0, 0]

    self.unwrapped.set_state(qpos, qvel)
