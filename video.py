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
# This file was branched from: https://github.com/facebookresearch/drqv2

import cv2
import imageio
import numpy as np


class VideoRecorder:
  def __init__(self, root_dir, camera_name, render_size=256, fps=20):
    if root_dir is not None:
      self.save_dir = root_dir / 'eval_video'
      self.save_dir.mkdir(exist_ok=True)
    else:
      self.save_dir = None

    self.camera_name = camera_name
    self.render_size = render_size
    self.fps = fps
    self.frames = []

  def init(self, env, enabled=True):
    self.frames = []
    self.enabled = self.save_dir is not None and enabled
    self.record(env)

  def record(self, env):
    if self.enabled:
      if hasattr(env, 'physics'):
        frame = env.physics.render(height=self.render_size,
                                   width=self.render_size,
                                   camera_id=0)
      else:
        render_kwargs = dict(
            offscreen=True,
            resolution=(self.render_size, self.render_size),
            camera_name=self.camera_name,
        )
        frame = env.render(**render_kwargs)
      self.frames.append(frame)

  def save(self, file_name):
    if self.enabled:
      path = self.save_dir / file_name
      imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
  def __init__(self, root_dir, render_size=256, fps=20):
    if root_dir is not None:
      self.save_dir = root_dir / 'train_video'
      self.save_dir.mkdir(exist_ok=True)
    else:
      self.save_dir = None

    self.render_size = render_size
    self.fps = fps
    self.frames = []

  def init(self, obs, enabled=True):
    self.frames = []
    self.enabled = self.save_dir is not None and enabled
    self.record(obs)

  def record(self, obs):
    if self.enabled:
      frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                         dsize=(self.render_size, self.render_size),
                         interpolation=cv2.INTER_CUBIC)
      self.frames.append(frame)

  def save(self, file_name):
    if self.enabled:
      path = self.save_dir / file_name
      imageio.mimsave(str(path), self.frames, fps=self.fps)
