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

"""Copies relevant files from `mujoco_scanned_objects` to an output directory.

python copy_mujoco_scanned_objects.py \
    --input-dir /path/to/mujoco_scanned_objects
    --output-dir /path/to/output
"""
import argparse
import glob
import os
import shutil


FILES_TO_COPY = [
    "*.stl",
    'asset.xml',
    "dependencies.xml",
    "body.xml",
    "texture.png",
]


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run sample test of one specific environment!')
  parser.add_argument('--input-dir', type=str,
                      default='/tmp/mujoco_scanned_objects',
                      help='Path to mujoco_scanned_objects repo')
  parser.add_argument('--output-dir', type=str,
                      default='/tmp/mujoco_scanned_objects',
                      help='Output directory')
  args = parser.parse_args()

  OBJECT_NAMES = [
      os.path.basename(x) for x in glob.glob(
          os.path.join(args.input_dir, 'models/*'))
      if os.path.isdir(x)]

  # Get object directories
  object_dirs = []
  for object_name in OBJECT_NAMES:
    object_dir = os.path.join(args.input_dir, "models", object_name)
    if not os.path.isdir(object_dir):
      print(f"WARNING: Object {object_name} not found. Did you mean:")
      for result in glob.glob(object_dir + '*'):
        print(result.split('/')[-1])
      continue

    # Create output directory if not exist
    output_object_dir = os.path.join(args.output_dir, object_name)
    os.makedirs(output_object_dir, exist_ok=True)

    # Copy files
    for filename in FILES_TO_COPY:
      paths = glob.glob(os.path.join(object_dir, filename))
      for path in paths:
        basename = os.path.basename(path)
        output_path = os.path.join(output_object_dir, basename)
        shutil.copy(path, output_path)
