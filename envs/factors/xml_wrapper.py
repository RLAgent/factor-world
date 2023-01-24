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

"""Base class for environments that need to reload XML."""
import abc
from typing import Any, Dict

from gym import spaces
from gym.utils import seeding
import numpy as np

from envs.factors.factor_wrapper import FactorWrapper


class XmlWrapper(FactorWrapper):
  def __init__(self,
               env_cls,
               env_kwargs: Dict[str, Any],
               factor_space: spaces.Box,
               init_factor_value: np.ndarray = None,
               num_resets_per_randomize: int = 1,
               seed: int = 0
               ):
    """Wrapper that generates a new XML file from factor value being set.

    Args:
      env_cls: metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env.SawyerXYZEnv
      env_kwargs: Arguments for env_cls.
      factor_space: Space where factors will be sampled from to generate XML.
      init_factor_value: Initial factor value (optional).
      num_resets_per_randomize: Number of resets() called until factors
        are randomized again.
      seed: Random seed.
    """
    self._env_cls = env_cls
    self._base_xml_path = (env_kwargs['model_name']
                           if 'model_name' in env_kwargs
                           else env_cls.MODEL_NAME)
    self._xml_path = self._base_xml_path
    self._env_kwargs = env_kwargs
    self.seed(seed)

    self.env = None
    self._prev_value = None
    self._reinitialize_env_with_xml(self._xml_path)
    super().__init__(
        env=self.env,
        factor_space=factor_space,
        init_factor_value=init_factor_value,
        num_resets_per_randomize=num_resets_per_randomize)

  @abc.abstractmethod
  def _generate_xml(self, value):
    """Generates XML from the factor value and returns the filepath."""

  def _reinitialize_env_with_xml(self, xml_path):
    if self.env is not None:
      self.env.close()

    env_kwargs = self._env_kwargs.copy()
    env_kwargs.update(model_name=xml_path)

    self.env = self._env_cls(**env_kwargs)
    self.env.reset_model()

  def _set_to_factor(self, value):
    # Avoid re-loading XML if value hasn't changed.
    if value == self._prev_value:
      print("WARNING(XmlWrapper): Factor value did not change. "
            "Not generating new XML.")
      return

    # Generate new XML from value.
    self._xml_path = self._generate_xml(value)
    self._prev_value = value

    # Re-initialize environment with the new XML.
    self._reinitialize_env_with_xml(self._xml_path)

  def seed(self, seed: int):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
