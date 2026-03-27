# MolmoSpaces-Isaac

This package provides functionality to load objects and scenes from the `MolmoSpaces`
ecosystem into [`IsaacSim`][0] and [`IsaacLab`][1], as well as the converters used to
generate the assets and houses from MuJoCo to USD format.

---
**Updates 🤖**
- **[2026/02/11]** : Code for converting assets and scenes from `MolmoSpaces` in `mjcf` format
into `usd` format that can be loaded into `IsaacSim`, or used with `IsaacLab` (will upload some
samples scripts for `IsaacLab`, as the collision groups are reverted by default when using IsaacLab's
InteractiveScene).
---


## Installation

Just install it using your package manager of choice (will grab all required dependencies,
including `IsaacSim-5.1.0` and `IsaacLab-2.3.1`):

⚠️ **NOTE**: Make sure you change directories to this package first, or you could
get some issues when trying to install it from the root repository. After installation,
you can go back to the root of the repo, as all the following commands will assume you
are at the root of the repo.

```
# If using `conda`, just use `pip` to install it
pip install -e .[dev,sim]

# If using `uv`, use `pip` as well
uv pip install -e .[dev,sim]
```

## Download the assets and scenes

We have a helper script `ms-download` that can be used to grab the desired assets and
scenes datasets in `usd` format, ready for use in `IsaacSim` and `IsaacLab`.

- To get the assets for a specific dataset (e.g. `thor`, `objaverse`):

```bash
ms-download --type usd --install-dir assets/usd --assets thor
```

This should have installed the `thor` assets into a cache directory at `$HOME/.molmospaces/usd/objects/thor`,
and then symlinked the correct version into the provided folder (in this case, at `ROOT-OF-REPO/assets/usd/objects/thor`).

You can then open an asset in `IsaacSim` by just dragging and dropping the `usd` file
into the editor. For example, below we show the `Fridge_1_mesh.usda` asset:

![gif-fridge-isaacsim][2]

- To get the scenes for a specific dataset (e.g. `ithor`, `procthor-10k-train`, etc.):

```bash
ms-download --type usd --install-dir assets/usd --scenes ithor
```

This should have installed the `ithor` and `procthor-10k-train` scenes into a cache directory at
`$HOME/.molmospaces/usd/scenes/ithor` and `$HOME/.molmospaces/usd/scenes/procthor-10k-train`
respectively, and then symlinked the correct version into the provided folder (in this case, at
`ROOT-OF-REPO/assets/usd/scenes/{ithor,procthor-10k-train}`).

You can then open a scene in `IsaacSim` by just dragging and dropping the `usd` file
into the editor. For example, below we show the `scene.usda` associated with the `FloorPlan1`
scene from the `ithor` dataset:

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

![gif-fridge-isaacsim][3]

[0]: <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html> (isaacsim-website)
[1]: <https://isaac-sim.github.io/IsaacLab/main/index.html> (isaaclab-website)
[2]: <https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbDlzdGJnZmM3amowaXh4ejZ0Z3hsb3Boc3BxenU3OTgzYWY2aTFtdSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uff0R6JFMYz4BnvD35/giphy.gif> (asset-isaacsim)
[3]: <https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExeG5zaXFsc3F2Zm9pY2h3aG16c3E3Z21ucHZrdThuZjJmcHd2aGU1MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LNmIRbG16EpiyVRsfN/giphy.gif> (scene-isaacsim)
