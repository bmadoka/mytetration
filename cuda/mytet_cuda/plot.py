"""Picture path: the original plotting lines, unchanged except that the figure is closed afterwards."""
import os
from pathlib import Path

# keep matplotlib's config/cache inside the project folder unless the caller chose a place
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


def png_name(variant, x0, y0, eps):
    if variant.endswith("_R"):
        return f"mytetration_x_{x0}_y_{y0}_eps_{eps}_rotated.png"
    return f"mytetration_x_{x0}_y_{y0}_eps_{eps}.png"


def save_png(divergence_map, variant, x0, y0, eps, eps_y, out_dir="."):
    """Write the PNG exactly as the original script does and return its path."""
    if eps_y is None:  # 1by1 files use eps for both axes
        eps_y = eps
    fig = plt.figure()
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["black", "white"])
    if variant.endswith("_R"):
        rotated_map = np.rot90(divergence_map, k=-1)
        plt.imshow(rotated_map.T, extent=[y0 - eps_y, y0 + eps_y, x0 - eps, x0 + eps], origin='lower', cmap=cmap)
    else:
        plt.imshow(divergence_map.T, extent=[x0 - eps, x0 + eps, y0 - eps_y, y0 + eps_y], origin='lower', cmap=cmap)
    plt.axis('off')
    path = os.path.join(out_dir, png_name(variant, x0, y0, eps))
    plt.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return path
