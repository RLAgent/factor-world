# About

This repository is already ready for use, and the instructions below do not need to be run. For reproducibility, we document how the asset files (XML, STL, PNG, etc.) were generated.

## Sample factors for train & eval

Randomly sample 81 textures each for `envs/assets/factors/factors_train.xml` and `envs/assets/factors/factors_eval.xml`:
```
python envs/scripts/generate_factors_xml.py
```

## Generate XML from Google Scanned Objects

This repo already contains Mujoco-compatible versions of the Google Scanned Objects dataset in `envs/assets/mujoco_sanned_objects`. Below, we document how we generated the XML and STL files.
```
cd /path/to/metaworld

# Clone the Google Scanned Objects dataset
git clone https://github.com/kevinzakka/mujoco_scanned_objects /tmp/mujoco_scanned_objects

# Create conda env
conda create -n scanned_objects python=3.10
conda activate scanned_objects

# Install dependencies
pip install meshio

# Convert STL to OBJ
python -m envs.scripts.stl_to_obj --path /tmp/mujoco_scanned_objects

# Generate XML files
python -m envs.scripts.generate_xml --path /tmp/mujoco_scanned_objects

# Copy files to this repo under metaworld/envs/mujoco_scanned_objects
python -m envs.scripts.copy_mujoco_scanned_objects \
    --input-dir /tmp/mujoco_scanned_objects \
    --output-dir $PWD/third_party/mujoco_scanned_objects/models

# Delete dataset
rm -rf /tmp/mujoco_scanned_objects
```

## Create symlink for metaworld

```
ln -s third_party/metaworld/metaworld metaworld
```
