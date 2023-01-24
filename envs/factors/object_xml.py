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

from typing import Any, Dict, Optional, Sequence, Tuple

from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation

from envs.factors import constants
from envs.factors.xml_utils import save_xml_with_object
from envs.factors.xml_wrapper import XmlWrapper


class ObjectXmlWrapper(XmlWrapper):
  def __init__(self,
               env_cls,
               env_kwargs,
               object_ids_range: Optional[Tuple[int, int]] = None,
               size_range: Tuple[float, float] = (0.3, 0.8),
               theta_range: Tuple[float, float] = (0, 2 * np.pi),
               init_factor_value: np.ndarray = None,
               num_resets_per_randomize: int = 1,
               seed: int = 0,
               ):
    """Adds a distractor object to the scene.

    Args:
        env_cls: Environment class
        xml_path: Base XML path
        seed: Random seed
        object_ids_range: Range of object IDs to sample from (inclusive). By
          default, use all object IDs.
        size_range: Visual size of the distractor objects (value between 0.0
          and 1.0)
        theta_range: Rotation of the distractor objects (value between 0 and
          2 * pi)
        env_kwargs: Kwargs for env initialization
    """
    assert size_range[0] <= size_range[1], size_range
    assert theta_range[0] <= theta_range[1], theta_range
    if object_ids_range is not None:
        assert object_ids_range[0] >= 0
        assert object_ids_range[0] <= object_ids_range[1]
        assert object_ids_range[1] < len(constants.OBJECT_NAMES), len(constants.OBJECT_NAMES)
    else:
      object_ids_range = (0, len(constants.OBJECT_NAMES))
    self._seed = seed

    super().__init__(
        env_cls,
        env_kwargs=env_kwargs,
        factor_space=spaces.Dict({
                'object_id': spaces.Discrete(
                    start=object_ids_range[0],
                    n=(object_ids_range[1] - object_ids_range[0])),
                'size': spaces.Box(
                    low=size_range[0], high=size_range[1], dtype=np.float32),
                'theta': spaces.Box(
                    low=theta_range[0], high=theta_range[1], dtype=np.float32),
            },
            seed=seed,
        ),
        init_factor_value=init_factor_value,
        num_resets_per_randomize=num_resets_per_randomize,
        seed=seed)

  def _generate_xml(self, value: Dict[str, Any]):
    self._object_name = constants.OBJECT_NAMES[value['object_id']]
    size = value['size'][0]
    theta = value['theta'][0]
    quat = Rotation.from_euler('x', theta).as_quat()

    xml_path = save_xml_with_object(
            self._base_xml_path,
            name=self._object_name,
            size=size,
            quaternion=quat)
    return xml_path

  @property
  def factor_name(self) -> str:
    return "object_xml"

  @property
  def object_name(self):
    return self._object_name
