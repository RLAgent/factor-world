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

"""Converts `mujoco_scanned_objects` XML files to be compatible with Metaworld.

For each scanned object, this script generates object.xml and dependencies.xml
from model.xml.

python data/generate_xml.py --path /path/to/mujoco_scanned_objects
"""
import argparse
import glob
import os
import xml.etree.ElementTree as ET

# Relative output paths must point:
#   from //third_party/metaworld/metaworld/envs/assets_v2/sawyer_xyz
#   to   //third_party/mujoco_scanned_objects/models
RELATIVE_OUTPUT_PATH = "../../../../../mujoco_scanned_objects/models/"

DEFAULT_OBJECT_MASS = 0.5


def append_str_to_attrib(x, attrib_key: str, val: str, delim='_'):
  """Sets `x.attrib[attrib_key]` to `x.attrib[attrib_key] + delim + val`."""
  if attrib_key in x.attrib:
    new_val = x.attrib[attrib_key] + delim + val
    x.set(attrib_key, new_val)
  else:
    # print(f"Warning: No attrib_key found: {attrib_key}")
    pass


def process_and_write(xml_path):
  # Output paths
  assert xml_path.endswith('model.xml')
  dependencies_xml_path = xml_path.replace('model.xml', 'dependencies.xml')
  body_xml_path = xml_path.replace('model.xml', 'body.xml')
  asset_xml_path = xml_path.replace('model.xml', 'asset.xml')

  name = xml_path.split('/')[-2]  # E.g., 'Android_Lego'
  print(f"Parsing {xml_path} for {name}")

  """Input XML Format:

  <mujoco model="model">
    <asset>
      ...
    </asset>
    <worldbody>
      <body name="model">
        ...
      </body>
    </worldbody>
  </mujoco>
  """
  tree = ET.parse(xml_path)
  root = tree.getroot()
  assert len(root) == 2, len(root)
  assert root[0].tag == 'asset'
  assert root[1].tag == 'worldbody'
  asset = root[0]
  worldbody = root[1]

  assert len(worldbody) == 1, len(worldbody)
  worldbody_children = worldbody[0]

  # ------------------- Build asset.xml -------------------
  for x in asset:
    if 'file' in x.attrib:
      # Rename filepaths to be relative
      new_file = os.path.join(RELATIVE_OUTPUT_PATH, name, x.attrib['file'])
      # Change filetype
      new_file = new_file.replace('.obj', '.stl')
      x.set('file', new_file)

    # Append object name, to make the names unique across all objects.
    append_str_to_attrib(x, 'name', name)
    append_str_to_attrib(x, 'texture', name)

    if x.tag == 'mesh':
      assert 'scale' not in x.attrib
      x.set('scale', "{scale} {scale} {scale}")  # Template value is set later

  # Write to file
  asset_tree = ET.ElementTree(asset)
  asset_tree.write(asset_xml_path)
  print(f"Saved to: {asset_xml_path}")

  # ------------------- Build body.xml -------------------
  body_xml = ET.fromstring(f"""
      <body childclass="{name}_base" name="{name}1">
        <geom class="{name}_col" name="{name}"
         type="mesh" mesh="model_{name}"
         mass="{DEFAULT_OBJECT_MASS}" friction="0.95 0.3 0.1"
         contype="1" conaffinity="1" condim="4"
         />
      </body>
      """)
  assert body_xml[0].tag == 'geom'
  body_xml[0].set('name', "{geom_name}")  # Template name to be set later

  # Append children
  for x in worldbody_children:
    append_str_to_attrib(x, 'mesh', name)
    append_str_to_attrib(x, 'material', name)
    body_xml.append(x)

  # Write to file
  body_tree = ET.ElementTree()
  body_tree._setroot(body_xml)
  body_tree.write(body_xml_path)
  print(f"Saved to: {body_xml_path}")

  # ------------------- Build dependencies.xml -------------------
  dependencies_xml = ET.fromstring(f"""
      <mujocoinclude>
        <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5"/>
        <default>
          <default class="{name}_base">
            <joint armature="0.001" damping="2" limited="true"/>
            <geom conaffinity="0" contype="0" group="1" type="mesh"/>
            <position ctrllimited="true" ctrlrange="0 1.57"/>
            <default class="{name}_col">
              <geom conaffinity="1" condim="4" contype="1" group="4"
               material="material_0_{name}" solimp="0.99 0.99 0.01"
               solref="0.01 1"/>
            </default>
          </default>
        </default>
      </mujocoinclude>
      """)

  # Write to file
  dependencies_tree = ET.ElementTree()
  dependencies_tree._setroot(dependencies_xml)
  dependencies_tree.write(dependencies_xml_path)
  print(f"Saved to: {dependencies_xml_path}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run sample test of one specific environment!')
  parser.add_argument('--path', type=str, default='/tmp/mujoco_scanned_objects',
                      help='Path to mujoco_scanned_objects repo')
  args = parser.parse_args()

  model_xml_regex = os.path.join(args.path, 'models/*/model.xml')
  input_xml_paths = glob.glob(model_xml_regex)

  for xml_path in input_xml_paths:
    process_and_write(xml_path)
