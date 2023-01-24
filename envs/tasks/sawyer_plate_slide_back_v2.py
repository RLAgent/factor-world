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

from gym.spaces import Box
import numpy as np
from scipy.spatial.transform import Rotation

from envs.asset_path_utils import full_v2_path_for
from envs.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from third_party.metaworld.metaworld.envs import reward_utils


class SawyerPlateSlideBackEnvV2(SawyerXYZEnv):
  MODEL_NAME = full_v2_path_for('sawyer_xyz/sawyer_plate_slide.xml')

  def __init__(self, model_name=MODEL_NAME, **kwargs):
    goal_low = (-0.1, 0.6, 0.015)
    goal_high = (0.1, 0.6, 0.015)
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    obj_low = (0., 0.85, 0.)
    obj_high = (0., 0.85, 0.)

    super().__init__(
        model_name,
        hand_low=hand_low,
        hand_high=hand_high,
        **kwargs,
    )

    self.init_config = {
        'obj_init_angle': 0.3,
        'obj_init_pos': np.array([0., 0.85, 0.], dtype=np.float32),
        'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
    }
    self.goal = np.array([0., 0.6, 0.015])
    self.obj_init_pos = self.init_config['obj_init_pos']
    self.obj_init_angle = self.init_config['obj_init_angle']
    self.hand_init_pos = self.init_config['hand_init_pos']

    self._random_reset_space = Box(
        np.hstack((obj_low, goal_low)),
        np.hstack((obj_high, goal_high)),
    )
    self.goal_space = Box(np.array(goal_low), np.array(goal_high))

  @_assert_task_is_set
  def evaluate_state(self, obs, action):
    (
        reward,
        tcp_to_obj,
        tcp_opened,
        obj_to_target,
        object_grasped,
        in_place
    ) = self.compute_reward(action, obs)

    success = float(obj_to_target <= 0.07)
    near_object = float(tcp_to_obj <= 0.03)

    info = {
        'success': success,
        'near_object': near_object,
        'grasp_success': 0.0,
        'grasp_reward': object_grasped,
        'in_place_reward': in_place,
        'obj_to_target': obj_to_target,
        'unscaled_reward': reward
    }
    return reward, info

  def _get_pos_objects(self):
    return self.data.get_geom_xpos('puck')

  def _get_quat_objects(self):
    return Rotation.from_matrix(self.data.get_geom_xmat('puck')).as_quat()

  def _set_obj_xyz(self, pos):
    qpos = self.data.qpos.flat.copy()
    qvel = self.data.qvel.flat.copy()
    qpos[9:11] = pos
    self.set_state(qpos, qvel)

  def reset_model(self):
    self._reset_hand()

    self.obj_init_pos = self.init_config['obj_init_pos']
    self._target_pos = self.goal.copy()

    if self.random_init:
      rand_vec = self._get_state_rand_vec()
      self.obj_init_pos = rand_vec[:3]
      self._target_pos = rand_vec[3:]

    self.sim.model.body_pos[self.model.body_name2id(
        'puck_goal')] = self.obj_init_pos
    self._set_obj_xyz(np.array([0, 0.15]))

    return self._get_obs()

  def compute_reward(self, actions, obs):
    _TARGET_RADIUS = 0.05
    tcp = self.tcp_center
    obj = obs[4:7]
    tcp_opened = obs[3]
    target = self._target_pos

    obj_to_target = np.linalg.norm(obj - target)
    in_place_margin = np.linalg.norm(self.obj_init_pos - target)
    in_place = reward_utils.tolerance(obj_to_target,
                                      bounds=(0, _TARGET_RADIUS),
                                      margin=in_place_margin - _TARGET_RADIUS,
                                      sigmoid='long_tail',)

    tcp_to_obj = np.linalg.norm(tcp - obj)
    obj_grasped_margin = np.linalg.norm(self.init_tcp - self.obj_init_pos)
    object_grasped = reward_utils.tolerance(tcp_to_obj,
                                            bounds=(0, _TARGET_RADIUS),
                                            margin=obj_grasped_margin - _TARGET_RADIUS,
                                            sigmoid='long_tail',)

    reward = 1.5 * object_grasped

    if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
      reward = 2 + (7 * in_place)

    if obj_to_target < _TARGET_RADIUS:
      reward = 10.
    return [
        reward,
        tcp_to_obj,
        tcp_opened,
        obj_to_target,
        object_grasped,
        in_place
    ]
