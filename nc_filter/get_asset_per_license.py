from molmo_spaces.utils.object_metadata import ObjectMeta

meta = ObjectMeta()
anno = list(meta.annotation().values())

from collections import Counter
from collections import defaultdict

license_to_asset = defaultdict(list)
c = Counter()
for a in anno:
    if "license_info" in a:
        c[a["license_info"]["license"]] += 1
        license_to_asset[a["license_info"]["license"]].append(a["assetId"])
    else:
        assert not a["isObjaverse"]
        license_to_asset["by"].append(a["assetId"])

print(c.most_common())
for l, assets in license_to_asset.items():
    print(l, len(assets))

import json
import gzip

with gzip.open("license_to_asset_id.json.gz", "wt") as f:
    json.dump({lic: sorted(assets) for lic, assets in license_to_asset.items()}, f)

