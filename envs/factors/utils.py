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

from typing import Any, Dict

from envs.asset_path_utils import full_assets_path_for
from envs.factors.arm_pos import ArmPosWrapper
from envs.factors.camera_pos import CameraPosWrapper
from envs.factors.distractor_pos import DistractorPosWrapper
from envs.factors.distractor_xml import DistractorXmlWrapper
from envs.factors.floor_texture import FloorTextureWrapper
from envs.factors.light import LightWrapper
from envs.factors.object_pos import ObjectPosWrapper
from envs.factors.object_size import ObjectSizeWrapper
from envs.factors.object_texture import ObjectTextureWrapper
from envs.factors.object_xml import ObjectXmlWrapper
from envs.factors.table_pos import TablePosWrapper
from envs.factors.table_texture import TableTextureWrapper
from envs.factors.xml_utils import get_texture_names
from envs.factors.xml_utils import generate_xml


FACTOR_NAME2CLS = {
    'arm_pos': ArmPosWrapper,
    'camera_pos': CameraPosWrapper,
    'distractor_pos': DistractorPosWrapper,
    'floor_texture': FloorTextureWrapper,
    'light': LightWrapper,
    'object_pos': ObjectPosWrapper,
    'object_size': ObjectSizeWrapper,
    'object_texture': ObjectTextureWrapper,
    'table_pos': TablePosWrapper,
    'table_texture': TableTextureWrapper,
}


def make_env_with_factors(
    env_cls, env_kwargs,
    factor_kwargs: Dict[str, Dict[str, Any]] = None,
    use_train_xml: bool = True,
):
  """Creates  that generates a new XML file from factor value being set.

  Args:
    env_cls: metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env.SawyerXYZEnv
    env_kwargs: Arguments for env_cls.
    factor_space: Space where factors will be sampled from to generate XML.
    use_train_xml: Whether to load train or eval XML (e.g., textures).
  """
  factors_relative_path = (
      "factors/factors_train.xml" if use_train_xml else
      "factors/factors_eval.xml")

  # Add includes to the XML, necessary for texture factors.
  base_xml_path = env_cls.MODEL_NAME
  # Relative path must point:
  #   from //third_party/metaworld/metaworld/envs/assets_v2/sawyer_xyz
  #   to   //envs/assets/mujoco_scanned_objects
  xml_path = generate_xml(
      base_xml_path,
      include_paths=["../../../../../../envs/assets/" + factors_relative_path])
  env_kwargs.update(model_name=xml_path)

  texture_names = get_texture_names(
      full_assets_path_for(factors_relative_path))

  # Cannot have both distractor_object and object factors, as both
  # require modifying the XML.
  assert not ('distractor_xml' in factor_kwargs and
              'object_xml' in factor_kwargs)
  if 'distractor_xml' in factor_kwargs:
    env = DistractorXmlWrapper(env_cls, env_kwargs,
                               **factor_kwargs['distractor_xml'])
    del factor_kwargs['distractor_xml']
  elif 'object_xml' in factor_kwargs:
    env = ObjectXmlWrapper(env_cls, env_kwargs,
                           **factor_kwargs['object_xml'])
    del factor_kwargs['object_xml']
  else:
    env = env_cls(**env_kwargs)

  texture_wrapper_seeds = []
  for factor_name, kwargs in factor_kwargs.items():
    # Need to copy omegaconf into a dict, in order to add new keys to dict.
    env_kwargs = {k: v for k, v in kwargs.items()}
    factor_cls = FACTOR_NAME2CLS[factor_name]
    if factor_name in ['floor_texture', 'object_texture', 'table_texture']:
      # By default, use all available textures.
      if 'texture_names' not in env_kwargs:
        env_kwargs.update(texture_names=texture_names)

      # Make sure seed is different across all texture wrappers.
      if 'seed' in env_kwargs:
        seed = env_kwargs['seed']
        while seed in texture_wrapper_seeds:
          seed += 1
        env_kwargs.update(seed=seed)
        texture_wrapper_seeds.append(seed)

    env = factor_cls(env, **env_kwargs)

  return env
