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
#
# This file was branched from: https://github.com/rlworkgroup/metaworld

import os


ENV_ASSET_DIR_V2 = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'third_party/metaworld/metaworld/envs/assets_v2')

MSO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'third_party/mujoco_scanned_objects/models')

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def full_v2_path_for(filename):
  return os.path.join(ENV_ASSET_DIR_V2, filename)


def full_assets_path_for(filename):
  return os.path.join(ASSETS_DIR, filename)


def full_mso_path_for(filename):
  return os.path.join(MSO_DIR, filename)
