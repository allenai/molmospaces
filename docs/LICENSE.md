# Dataset License

## 1. Default License

Unless otherwise specified, all assets, objects, grasps, and scenes created **in-house by the Allen Institute for AI (Ai2)** are licensed under:

**Creative Commons Attribution 4.0 International (CC BY 4.0)**
URL: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)

**Attribution requirements:**
When using these assets, please credit the Allen Institute for AI (Ai2) as the creator. For example:

> Model/Asset by the Allen Institute for AI, licensed under CC BY 4.0

**Derivative works and modifications:**
In-house assets may be modified (e.g., decimation, baking, procedural adjustments) to optimize performance or compatibility. Quality may differ from the original asset.

---

## 2. Third-party Assets

Some assets included in this dataset are from third parties and retain their original licenses. These may include:

- Robot models
- 3D object models modified from the originals in Sketchfab
- Other publicly-licensed datasets incorporated into scenes

**Important:** You must respect the original license of these third-party assets. Attribution, non-commercial restrictions, or share-alike clauses may apply depending on the source.

---

## 3. Retrieving per-asset license information

To retrieve the specific license for any asset in this dataset, you can use the provided helper function in your code environment:

```python
from molmo_spaces.molmo_spaces_constants import print_license_info

print_license_info(data_type, data_source, identifier)

"""
Parameters:
  data_type: One of "objects", "scenes", "grasps", "robots"
  data_source: Specific data source, e.g. "objaverse" for "objects"
  identifier: the unique identifier for the asset
"""
```

This will print the full license text and attribution for the selected asset. The comprehensive list of possible asset sources is:

```python
{
    "robots": {
        "rby1",
        "rby1m",
        "franka_droid",
        "floating_rum",
    },
    "scenes": {
        "ithor",
        "procthor-10k-train",
        "procthor-10k-val",
        "procthor-10k-test",
        "holodeck-objaverse-train",
        "holodeck-objaverse-val",
        "procthor-objaverse-train",
        "procthor-objaverse-val",
    },
    "objects": {
        "thor",
        "objaverse",
    },
    "grasps": {
        "droid",
        "droid_objaverse",
    },
}
```

For example, to read the license for a `holodeck-objaverse-train` scene:
```python
print_license_info("scenes", "holodeck-objaverse-train", 0)
```

or for an `objaverse` object:
```python
print_license_info("objects", "objaverse", "b8384089f301452783d8c7cf4778c23d")
```

The most general way to access license info is to provide an archive identifier. Possible archive names will be printed by e.g.:
```python
print_license_info("scenes", "ithor", "--list_all")
```