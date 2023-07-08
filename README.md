# Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation

This is the official codebase for:
```
@misc{xie2023decomposing,
    title={Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation},
    author={Xie*, Annie and Lee*, Lisa and Xiao, Ted and Finn, Chelsea},
    year={2023},
}
```

We provide MetaWorld environments with various factors of variation, including:
* Arm position
* Camera position
* Floor texture
* Lighting
* Object position
* Object size
* Object texture
* Table position
* Table texture
* Distractor objects & positions

# Preliminaries

Clone the repository:
```
git clone https://github.com/googleprivate/factor-world.git
cd factor-world
```

Create conda env:
```
conda env create -f conda_env.yml

conda activate factor_world

# Takes a while to run the first time
python -c "import mujoco_py"
```

Install Torch with CUDA support:
```
# Uninstall any previous versions of Torch.
conda uninstall -y pytorch
pip uninstall -y pytorch torchaudio torchvision

# NOTE: There is an issue (https://stackoverflow.com/a/71593841)
# where installing torchaudio and torchvision with pytorch
# in the same line causes the cpu-only versions to be installed.
# Install them separately by running:
conda install -y pytorch cudatoolkit=11.3 -c pytorch
conda install -y torchaudio torchvision -c pytorch

conda install -y pytorch==1.12.0 cudatoolkit=11.3 -c pytorch
conda install -y torchaudio==0.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch

# Check if CUDA versions of Torch have been installed
python -c "import torch; print(torch.cuda.is_available())"
conda list | grep torch
```

# Behavioral Cloning

To collect data into the output dir `/tmp/data`, modify the config in `cfgs/data.yaml` as neeed, then run:
```
bash run_scripted_policy_pick_place.sh $HOME/data
bash run_scripted_policy.sh $HOME/data  # TODO(lslee): Fix this script
```

To train on the data saved under `/tmp/data`, and log output to `/tmp/log`, modify the config in `cfgs/train_bc.yaml` as needed, then run:
```
bash train_bc.sh $HOME/data $HOME/log
```

# Environment visualization

Test and visualize all factor wrappers by running:
```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so:/usr/lib/x86_64-linux-gnu/libGLEW.so \
python -m factor_wrapper_test \
  --render=True
```

Run scripted policy and render to screen:
```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so:/usr/lib/x86_64-linux-gnu/libGLEW.so \
python -m run_scripted_policy \
    mode=render \
    num_episodes=100 \
    num_episodes_per_randomize=1 \
    seed=0 \
    factors=[arm_pos,light,object_size,object_texture,table_pos,table_texture,distractor_xml,distractor_pos] \
    task_name=pick-place-v2 \
    debug=True
```

To save a video into the output dir `/tmp/data`:
```
LD_PRELOAD='' \
python -m run_scripted_policy \
    mode=save_video \
    output_dir=/tmp/data \
    num_episodes=10 \
    num_episodes_per_randomize=1 \
    seed=0 \
    factors=[arm_pos,light,object_size] \
    task_name=basketball-v2
```

## List of supported environments
The flag value of `task_name` can be replaced with the following list of supported environments:
```
pick-place-v2
basketball-v2
bin-picking-v2
button-press-v2
button-press-topdown-v2
button-press-topdown-wall-v2
button-press-wall-v2
door-lock-v2
door-open-v2
door-unlock-v2
drawer-close-v2
drawer-open-v2
faucet-close-v2
faucet-open-v2
handle-press-v2
handle-pull-v2
handle-pull-side-v2
lever-pull-v2
window-close-v2
window-open-v2
```
# Acknowledgements

This repository builds upon the following codebases:
* Metaworld: https://github.com/rlworkgroup/metaworld
* Weakly Supervised Control: https://github.com/google-research/weakly_supervised_control
* Mujoco Scanned Objects: https://github.com/kevinzakka/mujoco_scanned_objects
* DrQ-v2: https://github.com/facebookresearch/drqv2
