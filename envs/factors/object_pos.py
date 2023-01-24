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

from envs.factors.factor_wrapper import FactorWrapper


class ObjectPosWrapper(FactorWrapper):
  """Wrapper over MuJoCo environments that modifies object position."""

  def __init__(self,
               env: gym.Env,
               x_range: Tuple[float, float] = (-0.3, 0.3),
               y_range: Tuple[float, float] = (-0.1, 0.2),
               z_range: Tuple[float, float] = (-0, 0),
               theta_range: Tuple[float, float] = (0, 2 * np.pi),
               seed: int = None,
               **kwargs):
    """Creates a new wrapper."""
    super().__init__(
        env,
        factor_space=spaces.Box(
            low=np.array([x_range[0], y_range[0], z_range[0], theta_range[0]]),
            high=np.array(
                [x_range[1], y_range[1], z_range[1], theta_range[1]]),
            dtype=np.float32,
            seed=seed),
        **kwargs)

    if hasattr(self, 'object_name'):
      joint_name = f"joint_{self.object_name}"
    else:
      joint_name = 'objjoint'

    if joint_name not in self.model.joint_names:
      print(f"WARNING(object_pos): Joint {joint_name} not found.")
      self.object_init_pos = None
      self.object_init_quat = None
    else:
      # Store object qpos/qvel indices
      self.i_qp = self.model.get_joint_qpos_addr(joint_name)[0]
      self.i_qv = self.model.get_joint_qvel_addr(joint_name)[0]

      qpos = self.data.qpos.flat.copy()
      self._default_init_pos = self.unwrapped.init_config['obj_init_pos']
      self._default_init_quat = qpos[self.i_qp + 3:self.i_qp + 7]

      self.object_init_pos = self._default_init_pos.copy()
      self.object_init_quat = self._default_init_quat.copy()

  def reset(self, force_randomize_factor: bool = False):
    super().reset(force_randomize_factor=force_randomize_factor)

    # Reset object pos.
    self._set_object_pos(
        self.object_init_pos,
        self.object_init_quat)

    return self.unwrapped._get_obs()

  @property
  def factor_name(self):
    return 'object_pos'

  def _set_to_factor(self, value: Tuple[float, float, float, float]):
    """Sets to the given factor."""
    if (not hasattr(self, '_default_init_pos') or
            not hasattr(self, '_default_init_quat')):
      print(
          "WARNING(object_pos): Missing _default_init_pos. Not setting factor.")
      return

    self.object_init_pos = self.unwrapped.init_config['obj_init_pos'] + value[:3]

    delta_radians = value[3]
    radians = Rotation.from_quat(self._default_init_quat).as_euler('xyz')
    radians[0] += delta_radians
    self.object_init_quat = Rotation.from_euler('xyz', radians).as_quat()

  def _set_object_pos(self, pos: np.ndarray, quat: np.ndarray):
    if not hasattr(self, 'i_qp'):
      print("WARNING(object_pos): Missing i_qp. Not setting object_pos.")
      return

    assert pos.shape == (3,), pos.shape
    assert quat.shape == (4,), quat.shape
    qpos = self.data.qpos.flat.copy()
    qvel = self.data.qvel.flat.copy()

    qpos[self.i_qp:self.i_qp + 3] = pos
    qpos[self.i_qp + 3:self.i_qp + 7] = quat
    qvel[self.i_qv:self.i_qv + 3] = [0, 0, 0]

    self.set_state(qpos, qvel)
