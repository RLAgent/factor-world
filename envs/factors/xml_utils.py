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

import glob
import os
from typing import List, Sequence, Tuple

from absl import logging
import xml.etree.ElementTree as ET

from envs.asset_path_utils import full_mso_path_for


def get_texture_names(xml_path: str) -> List[str]:
  """Returns texture names from an XML path.

  Args:
    xml_path: The XML path.
  Returns:
    A list of texure names.
  """
  tree = ET.parse(xml_path)
  root = tree.getroot()
  asset = root.find('asset')
  texture_names = []
  for e in asset.findall('texture'):
    texture_names.append(e.attrib['name'])
  return texture_names


def _save_xml(tree, output_path: str):
  """Writes XML to file."""
  if os.path.exists(output_path):
    logging.warning(f"XML file already exists: {output_path}")
  else:
    save_dir = os.path.dirname(output_path)
    os.makedirs(save_dir, exist_ok=True)
    logging.info('Saving XML to: {}'.format(output_path))
    tree.write(output_path)


def generate_xml(
        base_xml_path: str,
        include_paths: List[str]) -> str:
  """Generates XML and saves to file.

  Args:
    base_xml_path: The base XML file to modify.
    include_paths: Include statements to add to the base XML.
  Returns:
    Output XML path
  """
  assert base_xml_path.endswith('.xml')

  tree = ET.parse(base_xml_path)
  root = tree.getroot()
  assert root.tag == 'mujoco', root.tag

  # Add includes to the XML.
  xml_key = ''
  for include_path in include_paths:
    root.append(ET.fromstring(f"<include file=\"{include_path}\"/>"))
    xml_key += include_path.replace('.', '').replace('/', '')

  # Add movable camera
  worldbody = root.find('worldbody')
  worldbody.append(ET.fromstring(
      "<body name=\"tablelink_center\" pos=\"0 .6 0\"></body>"))
  worldbody.append(ET.fromstring(
      "<camera name=\"movable\" mode=\"targetbody\" target=\"tablelink_center\" pos=\"0 0.5 1.5\"/>"))

  # Save XML to file.
  xml_path = base_xml_path.replace('.xml', f'_{xml_key}.xml')
  _save_xml(tree, xml_path)

  return xml_path


def generate_xml_include(object_name: str,
                         scale: float = None) -> str:
  """Returns XML strings for <include/> and <asset/> elements.

  Args:
    object_name: Object name
    scale: Visual size of the object (value between 0.0 and 1.0)
  """
  if scale is None:
    scale = 1.0

  # Relative path should point:
  #   from //third_party/metaworld/metaworld/envs/assets_v2/sawyer_xyz
  #   to   //third_party/mujoco_scanned_objects/models
  INCLUDE_TEMPLATE = """
      <include file="../../../../../mujoco_scanned_objects/models/{object_name}/dependencies.xml"/>
      """
  include = INCLUDE_TEMPLATE.format(object_name=object_name)

  with open(full_mso_path_for(f'{object_name}/asset.xml'), 'r') as f:
    asset_template = f.read()
  asset = asset_template.format(scale=scale)

  return include, asset


def generate_xml_body(object_name: str,
                      pos: Tuple[float, float, float] = None,
                      quat: Tuple[float, float, float, float] = None,
                      body_name: str = None,
                      geom_name: str = None) -> str:
  """Returns XML string for <body/> elements.

  Args:
    object_name: Object names
    pos: Position of the objects
    quat: Quaternion of the objects
    body_name: Name of <body/> element. If None, use object_name by default.
    geom_name: Name of <geom/> element. If None, use object_name by default.
  """
  if pos is None:
    pos = [0., 0., 0.]
  if quat is None:
    quat = [1., 0., 0., 0.]
  if body_name is None:
    body_name = object_name
  if geom_name is None:
    geom_name = object_name

  OBJECT_BODY_TEMPLATE = """
        <body name="{body_name}" pos="{pos}" quat="{quat}">
            <freejoint name="joint_{object_name}"/>
            {body}
        </body>
        """

  with open(full_mso_path_for(f'{object_name}/body.xml'), 'r') as f:
    body_template = f.read()
  body = body_template.format(geom_name=geom_name)

  object_body = OBJECT_BODY_TEMPLATE.format(
      object_name=object_name,
      body_name=body_name,
      pos=f"{pos[0]} {pos[1]} {pos[2]}",
      quat=f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
      body=body,
  )

  return object_body


def _xml_key(object_names: Sequence[str],
             scales: Sequence[float],
             positions: Sequence[Tuple[float, float, float]],
             quaternions: Sequence[Tuple[float, float, float, float]]) -> str:
  keys = []
  for name, scale, pos, quat in zip(object_names, scales, positions, quaternions):
    # Take the first letter of each word.
    # Example: 'Utana_5_Porcelain_Ramekin_Large' -> 'U5PRL'
    key = ''.join([x[0] for x in name.split('_')])
    key += f'{scale:.1f}'
    # key += f':{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}'
    # key += f':{quat[0]:.1f},{quat[1]:.1f},{quat[2]:.1f},{quat[3]:.1f}'
    keys.append(key)
  return '-'.join(keys)


def get_path(base_xml_path: str, xml_key: str):
  """Appends xml_key to end of base_xml_path (before .xml)."""
  return base_xml_path.replace('.xml', f'_CACHE_{xml_key}.xml')


def _save_xml_with_distractors(
        base_xml_path: str,
        object_names: Sequence[str],
        scales: Sequence[float] = None,
        positions: Sequence[Tuple[float, float, float]] = None,
        quaternions: Sequence[Tuple[float, float, float, float]] = None) -> str:
  """Generates XML with the given objects, and writes to file.

  Returns:
    The XML filepath.
  """
  tree = ET.parse(base_xml_path)
  root = tree.getroot()
  worldbody = root.find('worldbody')

  # Add <include/> & <asset/> elements
  if scales is None:
    scales = [1.0] * len(object_names)
  for object_name, scale in zip(object_names, scales):
    include, asset = generate_xml_include(object_name, scale)
    root.append(ET.fromstring(include))
    root.append(ET.fromstring(asset))

  # Add <body/> elements
  for object_name, pos, quat in zip(object_names, positions, quaternions):
    body = generate_xml_body(object_name, pos, quat)
    worldbody.append(ET.fromstring(body))

  # Write XML to file.
  xml_key = _xml_key(object_names, scales, positions, quaternions)
  xml_path = get_path(base_xml_path, xml_key)
  _save_xml(tree, xml_path)

  return xml_path


def save_xml_with_distractors(
    base_xml_path: str,
    names: List[str],
    sizes: List[float],
    positions: List[Tuple[float, float, float]],
    quaternions: List[Tuple[float, float, float, float]],
    delete_old_cache: bool = True,
) -> str:
  """Modifies XML file to include distractor objects, and
  writes to file. Returns the XML filepath.

  Args:
      base_xml_path: XML file to modify
      names: Names of distractor objects
      sizes: Sizes of distractor objects
      positions: Initial positions of distractor objects
      quaternions: Initial quaternions of distractor objects
      delete_old_cache: Whether to delete existing XML cache files.
  """
  if delete_old_cache:
    cache_paths = glob.glob(get_path(base_xml_path, "*"))
    for path in cache_paths:
      os.remove(path)
      print(f"Deleted cached XML: {path}")

  # Generate XML
  xml_path = _save_xml_with_distractors(
      base_xml_path,
      object_names=names,
      scales=sizes,
      positions=positions,
      quaternions=quaternions)

  return xml_path


def save_xml_with_object(
    base_xml_path: str,
    name: str,
    size: float,
    quaternion: Tuple[float, float, float, float],
    delete_old_cache: bool = True,
) -> str:
  """Modifies XML file to change the type of the task object, and
  writes to file. Returns the XML filepath.

  Args:
      base_xml_path: XML file to modify
      num_object_types: Number of object types to sample from
      scale_range: Visual size of the distractor objects (value between 0.0 and 1.0)
      theta_range: Rotation of the distractor objects (value between 0 and 2 * pi)
      delete_old_cache: Whether to delete existing XML cache files.
      seed: Random seed
  """
  if delete_old_cache:
    cache_paths = glob.glob(get_path(base_xml_path, "*"))
    for path in cache_paths:
      os.remove(path)
      print(f"Deleted cached XML: {path}")

  xml_path = _save_xml_with_object(
      base_xml_path,
      object_name=name,
      scale=size,
      pos=None,  # Get position from base_xml_path
      quat=quaternion)

  return xml_path


def _save_xml_with_object(
        base_xml_path: str,
        object_name: str,
        scale: float = None,
        pos: Tuple[float, float, float] = None,
        quat: Tuple[float, float, float, float] = None) -> str:
  tree = ET.parse(base_xml_path)
  root = tree.getroot()
  worldbody = root.find('worldbody')

  # Find & remove object from XML
  xml_obj_pos = None
  for body in worldbody.findall('body'):
    for geom in body.findall('geom'):
      if 'name' in geom.attrib and geom.attrib['name'] == 'objGeom':
        # Store object position
        xml_obj_pos = body.attrib['pos']
        worldbody.remove(body)
        break
  assert xml_obj_pos is not None

  # By default, use the object position from XML.
  if pos is None:
    pos = [float(x) for x in xml_obj_pos.split(' ')]

  # Add <include/> & <asset/> elements
  include, asset = generate_xml_include(object_name,
                                        scale=scale)
  root.append(ET.fromstring(include))
  root.append(ET.fromstring(asset))

  # Add <body/> element
  body = generate_xml_body(object_name, pos, quat,
                           body_name='obj',
                           geom_name='objGeom')
  worldbody.append(ET.fromstring(body))

  # Write XML to file.
  xml_key = _xml_key([object_name], [scale], [pos], [quat])
  xml_path = get_path(base_xml_path, xml_key)
  _save_xml(tree, xml_path)

  return xml_path
