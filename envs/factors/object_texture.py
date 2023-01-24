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

from typing import List

import gym
from gym import spaces
import mujoco_py
import numpy as np

from envs.factors.factor_wrapper import FactorWrapper


class ObjectTextureWrapper(FactorWrapper):
  """Wrapper over MuJoCo environments that modifies object texture."""

  def __init__(self,
               env: gym.Env,
               texture_names: List[str],
               seed: int = None,
               **kwargs):
    """Creates a new wrapper.

    Args:
      env: Environment to wrap
      texture_names: Texture names to sample from
      seed: Random seed
    """
    self.texture_names = texture_names

    super().__init__(
        env,
        factor_space=spaces.Discrete(
            start=0,
            n=len(self.texture_names),
            seed=seed),
        init_factor_value=0,
        **kwargs)

  @property
  def factor_name(self):
    return 'object_texture'

  def _set_to_factor(self, value: int):
    """Sets to the given factor."""
    if isinstance(value, np.ndarray):
      assert value.dtype == np.int32
      assert value.shape == (1,)
      value = value.item()

    tex_id = self._get_tex_id(value)

    # Make sure texture is different from table texture.
    table_mat_id = mujoco_py.cymj._mj_name2id(
        self.model, mujoco_py.cymj._mju_str2Type('material'), 'table_wood')
    while tex_id == self.model.mat_texid[table_mat_id]:
      value = (value + 1) % len(self.texture_names)
      tex_id = self._get_tex_id(value)

    if hasattr(self, 'object_name'):
      mat_name = f'material_0_{self.object_name}'
    else:
      mat_name = 'block_wood'
    mat_id = mujoco_py.cymj._mj_name2id(
        self.model, mujoco_py.cymj._mju_str2Type('material'), mat_name)
    assert mat_id != -1, mat_name

    # Set texture of object
    self.unwrapped.model.mat_texid[mat_id] = tex_id

  def _get_tex_id(self, value: int):
    texture_name = self.texture_names[value]
    return self._texture_name2id(texture_name)

  def _texture_name2id(self, texture_name: str):
    return mujoco_py.cymj._mj_name2id(
        self.model, mujoco_py.cymj._mju_str2Type('texture'), texture_name)
