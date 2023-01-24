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


class SawyerReachWallEnvV2(SawyerXYZEnv):
  """
  Motivation for V2:
      V1 was difficult to solve since the observations didn't say where
      to move (where to reach).
  Changelog from V1 to V2:
      - (7/7/20) Removed 3 element vector. Replaced with 3 element position
          of the goal (for consistency with other environments)
      - (6/17/20) Separated reach from reach-push-pick-place.
      - (6/17/20) Added a 3 element vector to the observation. This vector
          points from the end effector to the goal coordinate.
          i.e. (self._target_pos - pos_hand)
  """
  MODEL_NAME = full_v2_path_for('sawyer_xyz/sawyer_reach_wall_v2.xml')

  def __init__(self, model_name=MODEL_NAME, **kwargs):
    goal_low = (-0.05, 0.85, 0.05)
    goal_high = (0.05, 0.9, 0.3)
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    obj_low = (-0.05, 0.6, 0.015)
    obj_high = (0.05, 0.65, 0.015)

    super().__init__(
        model_name,
        hand_low=hand_low,
        hand_high=hand_high,
        **kwargs,
    )

    self.init_config = {
        'obj_init_angle': .3,
        'obj_init_pos': np.array([0, 0.6, 0.02]),
        'hand_init_pos': np.array([0, .6, .2]),
    }

    self.goal = np.array([-0.05, 0.8, 0.2])

    self.obj_init_angle = self.init_config['obj_init_angle']
    self.obj_init_pos = self.init_config['obj_init_pos']
    self.hand_init_pos = self.init_config['hand_init_pos']

    self._random_reset_space = Box(
        np.hstack((obj_low, goal_low)),
        np.hstack((obj_high, goal_high)),
    )
    self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    self.num_resets = 0

  @_assert_task_is_set
  def evaluate_state(self, obs, action):

    reward, tcp_to_object, in_place = self.compute_reward(action, obs)
    success = float(tcp_to_object <= 0.05)

    info = {
        'success': success,
        'near_object': 0.,
        'grasp_success': 0.,
        'grasp_reward': 0.,
        'in_place_reward': in_place,
        'obj_to_target': tcp_to_object,
        'unscaled_reward': reward,
    }

    return reward, info

  def _get_pos_objects(self):
    return self.get_body_com('obj')

  def _get_quat_objects(self):
    return Rotation.from_matrix(
        self.data.get_geom_xmat('objGeom')
    ).as_quat()

  def reset_model(self):
    self._reset_hand()
    self._target_pos = self.goal.copy()
    self.obj_init_angle = self.init_config['obj_init_angle']

    if self.random_init:
      goal_pos = self._get_state_rand_vec()
      self._target_pos = goal_pos[3:]
      while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
      self._target_pos = goal_pos[-3:]
      self.obj_init_pos = goal_pos[:3]

    self._set_obj_xyz(self.obj_init_pos)
    self.num_resets += 1

    return self._get_obs()

  def compute_reward(self, actions, obs):
    _TARGET_RADIUS = 0.05
    tcp = self.tcp_center
    obj = obs[4:7]
    tcp_opened = obs[3]
    target = self._target_pos

    tcp_to_target = np.linalg.norm(tcp - target)
    obj_to_target = np.linalg.norm(obj - target)

    in_place_margin = (np.linalg.norm(self.hand_init_pos - target))
    in_place = reward_utils.tolerance(tcp_to_target,
                                      bounds=(0, _TARGET_RADIUS),
                                      margin=in_place_margin,
                                      sigmoid='long_tail',)

    return [10 * in_place, tcp_to_target, in_place]
