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

from wrappers import make_wrapped_env
from wrappers import MetaWorldState
from third_party.metaworld.metaworld import policies
from replay_buffer import ReplayBufferStorage
import os

# import argparse
import cv2
import hydra
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple

# Hack to import submodule. Must run this script from root directory, e.g.,
#   python data/run_scripted_policy.py
import sys
sys.path.append('.')


POLICIES = {
    'assembly-v2': policies.SawyerAssemblyV2Policy,
    'basketball-v2': policies.SawyerBasketballV2Policy,
    'bin-picking-v2': policies.SawyerBinPickingV2Policy,
    'box-close-v2': policies.SawyerBoxCloseV2Policy,
    'button-press-v2': policies.SawyerButtonPressV2Policy,
    'button-press-topdown-v2': policies.SawyerButtonPressTopdownV2Policy,
    'button-press-topdown-wall-v2': policies.SawyerButtonPressTopdownWallV2Policy,
    'button-press-wall-v2': policies.SawyerButtonPressWallV2Policy,
    'coffee-button-v2': policies.SawyerCoffeeButtonV2Policy,
    'coffee-pull-v2': policies.SawyerCoffeePullV2Policy,
    'coffee-push-v2': policies.SawyerCoffeePushV2Policy,
    'dial-turn-v2': policies.SawyerDialTurnV2Policy,
    'disassemble-v2': policies.SawyerDisassembleV2Policy,
    'door-close-v2': policies.SawyerDoorCloseV2Policy,
    'door-lock-v2': policies.SawyerDoorLockV2Policy,
    'door-open-v2': policies.SawyerDoorOpenV2Policy,
    'door-unlock-v2': policies.SawyerDoorUnlockV2Policy,
    'drawer-close-v2': policies.SawyerDrawerCloseV2Policy,
    'drawer-open-v2': policies.SawyerDrawerOpenV2Policy,
    'faucet-close-v2': policies.SawyerFaucetCloseV2Policy,
    'faucet-open-v2': policies.SawyerFaucetOpenV2Policy,
    'hammer-v2': policies.SawyerHammerV2Policy,
    'hand-insert-v2': policies.SawyerHandInsertV2Policy,
    'handle-press-v2': policies.SawyerHandlePressV2Policy,
    'handle-pull-v2': policies.SawyerHandlePullV2Policy,
    'handle-pull-side-v2': policies.SawyerHandlePullSideV2Policy,
    'lever-pull-v2': policies.SawyerLeverPullV2Policy,
    'peg-insert-side-v2': policies.SawyerPegInsertionSideV2Policy,
    'pick-out-of-hole-v2': policies.SawyerPickOutOfHoleV2Policy,
    'pick-place-v2': policies.SawyerPickPlaceV2Policy,
    'pick-place-wall-v2': policies.SawyerPickPlaceWallV2Policy,
    'plate-slide-v2': policies.SawyerPlateSlideV2Policy,
    'plate-slide-back-v2': policies.SawyerPlateSlideBackV2Policy,
    'plate-slide-side-v2': policies.SawyerPlateSlideSideV2Policy,
    'push-v2': policies.SawyerPushV2Policy,
    'push-back-v2': policies.SawyerPushBackV2Policy,
    'push-wall-v2': policies.SawyerPushWallV2Policy,
    'reach-v2': policies.SawyerReachV2Policy,
    'shelf-place-v2': policies.SawyerShelfPlaceV2Policy,
    'soccer-v2': policies.SawyerSoccerV2Policy,
    'stick-pull-v2': policies.SawyerStickPullV2Policy,
    'stick-push-v2': policies.SawyerStickPushV2Policy,
    'sweep-into-v2': policies.SawyerSweepIntoV2Policy,
    'sweep-v2': policies.SawyerSweepV2Policy,
    'window-close-v2': policies.SawyerWindowCloseV2Policy,
    'window-open-v2': policies.SawyerWindowOpenV2Policy,
}


def _filename(cfg, image_obs_size: Tuple[int, int] = None):
  filename = f'{cfg.env.env_name}-{cfg.env.camera_name}'
  if image_obs_size:
    filename += f'-{image_obs_size[0]}x{image_obs_size[1]}'
  filename += f'-a{cfg.policy.action_noise}'
  if cfg.env.factors:
    filename += '-' + '-'.join(cfg.env.factors.keys())
  filename += f'-n{cfg.num_episodes}_{cfg.num_episodes_per_randomize}'
  return filename


def _get_image_obs_size(mode: str, image_obs_size):
  if mode == 'render':
    return None
  elif mode == 'save_video':
    return (600, 400)
  else:
    return image_obs_size


@hydra.main(config_path='cfgs', config_name='data')
def main(cfg):
  assert cfg.mode in ['render', 'save_video', 'save_buffer'], cfg.mode
  image_obs_size = _get_image_obs_size(cfg.mode, cfg.env.image_obs_size)

  factor_kwargs = {factor: cfg.env.factors[factor] for factor in cfg.factors}
  eval_factor_kwargs = {
      factor: cfg.env.eval_factors[factor] for factor in cfg.factors}

  # Create environment for collecting data.
  env = make_wrapped_env(
      cfg.env.env_name,
      use_train_xml=True,
      factor_kwargs=factor_kwargs,
      image_obs_size=image_obs_size,
      camera_name=cfg.env.camera_name,
      observe_goal=cfg.env.observe_goal,
      default_num_resets_per_randomize=cfg.env.num_resets_per_randomize)

  # Create environment for sampling factor values for evaluation only.
  eval_env = make_wrapped_env(
      cfg.env.env_name,
      use_train_xml=False,
      factor_kwargs=eval_factor_kwargs,
      image_obs_size=image_obs_size,
      camera_name=cfg.env.camera_name,
      observe_goal=cfg.env.observe_goal,
      default_num_resets_per_randomize=cfg.env.num_resets_per_randomize)

  # Get policy
  action_space_ptp = env.action_space.high - env.action_space.low
  noise = np.ones(env.action_space.shape) * cfg.policy.action_noise
  policy = POLICIES[cfg.task_name]()

  os.makedirs(cfg.output_dir, exist_ok=True)

  # Replay buffer output
  replay_buffer = None
  if cfg.mode == 'save_buffer':
    data_specs = {
        'observations': env.observation_space['image'],
        'states': MetaWorldState(env).observation_space['proprio'],
        'actions': env.action_space,
    }
    replay_buffer = ReplayBufferStorage(data_specs, Path(cfg.output_dir))

  # Video output
  video_writer = None
  if cfg.mode in ['save_video', 'save_buffer']:
    task_str = '%s:%s' % (
        cfg.task_name,
        '-'.join(cfg.factors))
    video_path = os.path.join(cfg.output_dir, f'video-{task_str}.avi')
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        env.metadata['video.frames_per_second'],
        image_obs_size)

  o = env.reset()
  data_factor_values = [env.current_factor_values]

  eval_env.reset()
  eval_factor_values = [eval_env.current_factor_values]

  num_successful_ep = 0
  num_failed_ep = 0
  while num_successful_ep < cfg.num_episodes:
    # Roll out an episode.
    ts = 0
    episode = []
    done = False
    while not done:
      # Sample action
      a = policy.get_action(o['proprio'])
      a = np.random.normal(a, noise * action_space_ptp)

      next_o, r, done, info = env.step(a)

      time_step = {
          'observations': o['image'],
          'states': MetaWorldState.state(o['proprio']),
          'actions': a.astype(np.float32),
          'rewards': np.array([r], dtype=np.float32),
          'discounts': np.array([1.0], dtype=np.float32),
      }
      episode.append(time_step)

      if cfg.mode == 'render':
        env.render()

      if video_writer:
        image = np.transpose(o['image'], (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image)

      o = next_o
      ts += 1

    # If unsuccessful, do not save episode.
    if ts >= env.max_path_length:
      num_failed_ep += 1
      if cfg.debug:
        print("Failed episode")
      o = env.reset_model()
    else:
      num_successful_ep += 1
      if cfg.debug:
        print("Successful episode")

      if replay_buffer is not None:
        replay_buffer.add_episode(episode)

      o = env.reset()
      data_factor_values.append(env.current_factor_values)

      eval_env.reset()
      eval_factor_values.append(eval_env.current_factor_values)

  print(f'Finished {num_successful_ep} episodes ({num_failed_ep} fails).')

  # Save factor values
  if cfg.mode == 'save_buffer':
    factors_pkl_path = os.path.join(cfg.output_dir, 'data_factor_values.pkl')
    with open(factors_pkl_path, 'wb') as fp:
      pickle.dump(data_factor_values, fp)

    factors_pkl_path = os.path.join(cfg.output_dir, 'eval_factor_values.pkl')
    with open(factors_pkl_path, 'wb') as fp:
      pickle.dump(eval_factor_values, fp)

  if video_writer:
    video_writer.release()


if __name__ == '__main__':
  main()
