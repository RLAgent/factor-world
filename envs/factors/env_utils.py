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

from gym import spaces
import numpy as np

from envs.factors import constants
from envs.sawyer_xyz_env import SawyerXYZEnv


def get_table_pos(env: SawyerXYZEnv) -> Tuple[float, float, float]:
  id = env.model.body_name2id('tablelink')
  return env.model.body_pos[id].copy()


def sample_object_positions_on_table(
        num_samples: int,
        table_pos: Tuple[float, float, float] = constants.DEFAULT_TABLE_POS,
        table_size: Tuple[float, float, float] = constants.DEFAULT_TABLE_SIZE,
        object_height: float = 0.0,
        edge_distance: float = 0.05,
        seed: int = None) -> np.ndarray:
  """Samples random object positions on the table.

  Args:
      num_samples: Number of positions to sample
      table_pos: Center position of the table
      table_size: Collision box size of the table
      object_height: Height of the object
      edge_distance: Min distance from edges of the table
      seed: Random seed
  Returns:
      An array of shape (n, 3)
  """
  pos_z = table_pos[2] + object_height
  pos_z = np.array([[pos_z] * num_samples]).transpose()  # (n, 1)

  xy_range_low = table_pos[:2] - table_size[:2] / 2 + edge_distance
  xy_range_high = table_pos[:2] + table_size[:2] / 2 - edge_distance

  pos_space = spaces.Box(
      low=np.array(xy_range_low.tolist() * num_samples),
      high=np.array(xy_range_high.tolist() * num_samples),
      dtype=np.float32,
      seed=seed)

  # TODO(lslee): Avoid sampling positions in the path between goal & object.
  pos_xy = pos_space.sample()
  pos_xy = pos_xy.reshape((num_samples, 2))
  pos = np.append(pos_xy, pos_z, axis=1)
  return pos
