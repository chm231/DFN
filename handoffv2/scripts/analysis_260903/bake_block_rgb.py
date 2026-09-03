# -*- coding: utf-8 -*-
"""블록 다면체 vtp에 블록별 고정 RGB 셀 배열을 굽는다 (tab20 mod 20).

사용: python bake_block_rgb.py <in_vtp> <out_vtp>
"""
import sys

import numpy as np
import pyvista as pv
from matplotlib import cm

m = pv.read(sys.argv[1])
bid = np.asarray(m.cell_data["block_id"])
tab = (np.array(cm.get_cmap("tab20").colors) * 255).astype(np.uint8)
m.cell_data["rgb"] = tab[bid % 20]
m.save(sys.argv[2])
print(f"baked rgb: {len(np.unique(bid))} blocks → {sys.argv[2]}")
