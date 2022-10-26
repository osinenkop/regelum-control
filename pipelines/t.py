import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


fig = plt.figure(figsize=(10, 10))

# test_rotation_mode(fig, "anchor", gs[1])
ax = fig.add_axes((-1, -1, 2, 2))
left, right = plt.xlim()
bottom, top = plt.ylim()
plt.xlim(left=-1)
plt.xlim(right=1)
plt.ylim(top=1)
plt.ylim(bottom=-1)
arrow = patches.FancyArrowPatch((0.5, 0.5), (0.3, 0.3))
arrow.set_positions(None, (0.6, 0.6))

ax.add_patch(arrow)

print(arrow.get_path())


plt.show()
