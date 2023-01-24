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

"""Converts .obj to .stl in the given `mujoco_scanned_objects` directory.

This script does not delete the original .obj files.

python generate_xml.py --path /path/to/mujoco_scanned_objects
"""
import argparse
import glob
import os

import meshio


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run sample test of one specific environment!')
  parser.add_argument('--path', type=str, default='/tmp/mujoco_scanned_objects',
                      help='Path to mujoco_scanned_objects repo')
  args = parser.parse_args()

  obj_regex = os.path.join(args.path, "models/*/*.obj")
  obj_files = glob.glob(obj_regex)
  print(f"Found {len(obj_files)} files.")

  for obj_file in obj_files:
    output_stl = obj_file.replace('.obj', '.stl')
    mesh = meshio.read(obj_file)
    mesh.write(output_stl, binary=True)
    print(f"Wrote: {output_stl}")
