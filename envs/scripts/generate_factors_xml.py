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

"""Randomly samples 81 textures each for XML files:
  //envs/assets/factors/factors_train.xml
  //envs/assets/factors/factors_eval.xml

Usage:
  python envs/scripts/generate_factors_xml.py
"""
import glob
import os

from lxml import etree
import numpy as np
import xml.etree.ElementTree as ET


TEXTURES_DIR = "envs/assets/textures/"

# Number of textures to sample per output path.
NUM_TEXTURES_PER_XML = 81

# Output paths
FACTORS_TRAIN_XML = "envs/assets/factors/factors_train.xml"
FACTORS_EVAL_XML = "envs/assets/factors/factors_eval.xml"

XML_TEMPLATE = """
<mujocoinclude><asset>
{textures}
</asset></mujocoinclude>"""

LICENSE_HEADER = """
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


def save_texture_includes_to_xml(textures, output_path: str):
  # File path must point:
  #   from //third_party/metaworld/metaworld/envs/assets_v2/sawyer_xyz
  #   to   //envs/assets/textures
  textures_xml = [
      f"  <texture name=\"tex_{texture}\" type=\"2d\" "
      f"file=\"../../../../../../envs/assets/textures/{texture}.png\" />"
      for texture in textures
  ]

  # Add license header
  license_header = etree.Comment(LICENSE_HEADER)
  root = etree.fromstring(XML_TEMPLATE.format(
      textures='\n'.join(textures_xml)))
  root.addprevious(license_header)
  tree = etree.ElementTree(root)

  tree.write(output_path)


if __name__ == '__main__':
  # Find texture files.
  texture_paths = glob.glob(os.path.join(TEXTURES_DIR, "*.png"))
  texture_names = [x.split('/')[-1].replace('.png', '') for x in texture_paths]
  print(f"Found {len(texture_names)} total textures.")

  # Sample textures.
  textures = np.random.choice(
      texture_names, size=NUM_TEXTURES_PER_XML * 2, replace=False)
  train_textures = textures[:NUM_TEXTURES_PER_XML]
  eval_textures = textures[NUM_TEXTURES_PER_XML:]

  save_texture_includes_to_xml(train_textures, FACTORS_TRAIN_XML)
  save_texture_includes_to_xml(eval_textures, FACTORS_EVAL_XML)
