# MolmoSpaces-ManiSkill

This package provides functionality to load objects and scenes from the MolmoSpaces
ecosystem into [`Sapien`][5] and [`ManiSkill`][6].

---
**Updates 🤖**
- **[2026/02/11]** : Code for loading assets and scenes from `MolmoSpaces` in `mjcf` format
into actors an   d articulations that can be used with a `Sapien` scene or a `ManiSkill` scene.
---


## Installation

Just install it using your package manager of choice (will grab all required dependencies,
including `maniskill` and `sapien`):

⚠️ **NOTE**: Make sure you change directories to this package first, or you could
get some issues when trying to install it from the root repository. After installation,
you can go back to the root of the repo, as all the following commands will assume you
are at the root of the repo.

```
# If using `conda`, just use `pip` to install it
pip install -e .[dev]

# If using `uv`, use `pip` as well
uv pip install -e .[dev]
```

⚠️ **NOTE**: For `MacOS` support, make sure you follow [these][4] instructions on how to
setup `Vulkan` on your system. The dependencies already install the `nightly` version of
ManiSkill, so just have to do the Vulkan setup.

## Download the assets and scenes

We have a helper script `ms-download` that can be used to grab the desired assets and
scenes datasets in `mjcf` format, which are the ones supported by our provided loaders.

- To get the assets for a specific dataset (e.g. `thor`, `objaverse`):

```bash
ms-download --type mjcf --install-dir assets/mjcf --assets thor
```

This should have installed the `thor` assets into a cache directory at `$HOME/.molmospaces/mjcf/objects/thor`,
and then symlinked the correct version into the provided folder (in this case, at `ROOT-OF-REPO/assets/mjcf/objects/thor`).

- To get the scenes for a specific dataset (e.g. `ithor`, `procthor-10k-train`, etc.):

```bash
ms-download --type mjcf --install-dir assets/mjcf --scenes ithor procthor-10k-train
```

This should have installed the `ithor` and `procthor-10k-train` scenes into a cache directory at
`$HOME/.molmospaces/mjcf/scenes/ithor` and `$HOME/.molmospaces/mjcf/scenes/procthor-10k-train`
respectively, and then symlinked the correct version into the provided folder (in this case, at
`ROOT-OF-REPO/assets/mjcf/scenes/{ithor,procthor-10k-train}`).

⚠️ **NOTE**: We're currently refactoring our resource manager, so we have three different
version provided for the three current simulators supported by `MolmoSpaces`. If you ran
the setup for the `MuJoCo` version of the package, from the `molmo_spaces` folder, the assets
and scenes will be saved in a different cache folder, and symlinked to the `assets` folder, instead
of the `assets/mjcf` folder. You can use those scenes just fine, just modify the path given to the
scripts accordingly.

## Examples

### Loading an asset

The example [`ex_mjcf_loader.py`][0] shows hou to load a simple mjcf asset from `MolmoSpaces`
into a `Sapien` scene, and show it in the viewer. Just run the following command:

```bash
python -m molmo_spaces_maniskill.examples.ex_mjcf_loader --filepath "assets/mjcf/objects/thor/Kitchen Objects/Fridge/Prefabs/Fridge_1/Fridge_1_prim.xml" \
    --mode articulation \
    --rotate
```

The viewer should be launched and you should be able to see the fridge in the viewer, like this:

![gif-asset-loader-example-fridge][1]

### Loading a scene

The example [`ex_scene_loader.py`][2] shows how to load a scene from `MolmoSpaces` into
a `Sapien` scene, and show it in the viewer. Just run the following command:

```bash
python -m molmo_spaces_maniskill.examples.ex_scene_loader --filepath assets/mjcf/scenes/procthor-10k-train/train_0.xml
```

The viewer should be launched and you should be able to see the scene in the viewer, like this:

![gif-scene-loader-example-1][3]

## Finding assets

To search assets of a specific type, we can just do

```python
from molmo_spaces.utils.object_retriever import ObjectRetriever
from molmo_spaces.utils.object_metadata import ObjectMeta

r = ObjectRetriever()
uids, sims = r.query("cellphone")
for it, (uid, sim) in enumerate(zip(uids, sims)):
  anno = ObjectMeta.annotation(uid)
  print(
      f"{it} {sim=} uid={uid} obja={anno['isObjaverse']} split={anno['split']} cat=`{anno['category']}`:"
      f" {anno['description_short']['five_words']}"
  )
```


## Citations

This package builds on top of both [`Sapien`][5] and [`ManiSkill`][6]:

```
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal = {Robotics: Science and Systems},
  year={2025},
}
```

```
@InProceedings{Xiang_2020_SAPIEN,
author = {Xiang, Fanbo and Qin, Yuzhe and Mo, Kaichun and Xia, Yikuan and Zhu, Hao and Liu, Fangchen and Liu, Minghua and Jiang, Hanxiao and Yuan, Yifu and Wang, He and Yi, Li and Chang, Angel X. and Guibas, Leonidas J. and Su, Hao},
title = {{SAPIEN}: A SimulAted Part-based Interactive ENvironment},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}}
```

[0]: <https://github.com/allenai/molmospaces/blob/main/molmo_spaces_maniskill/src/molmo_spaces_maniskill/examples/ex_mjcf_loader.py> (asset-loader-example)
[1]: <https://media.giphy.com/media/yk2FlMgSLw5fsUsYLC/giphy.gif> (asset-loader-example-gif)
[2]: <https://github.com/allenai/molmospaces/blob/main/molmo_spaces_maniskill/src/molmo_spaces_maniskill/examples/ex_scene_loader.py> (scene-loader-example)
[3]: <https://media.giphy.com/media/h4pFH8rC5HUgZcjjQj/giphy.gif> (scene-loader-example-gif)
[4]: <https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/macos_install.html> (maniskill-macos-install-instructions)
[5]: <https://github.com/haosulab/sapien> (sapien-github-repo)
[6]: <https://github.com/haosulab/maniskill> (maniskill-github-repo)
