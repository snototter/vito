import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vito import colormaps
from vito import imvis


def colormap_gradient(colormap_name: str, height: int = 20) -> np.ndarray:
    g = np.zeros((height, 256), dtype=np.uint8)
    for r in range(height):
        g[r, :] = [c for c in range(256)]
    return imvis.pseudocolor(g, limits=[0, 255], color_map=colormaps.by_name(colormap_name))


def display_colormaps():
    for cmn in colormaps.colormap_names:
        vis = colormap_gradient(cmn)
        imvis.imshow(vis, f'Colormap {cmn}', wait_ms=-1)


if __name__ == '__main__':
    display_colormaps()
