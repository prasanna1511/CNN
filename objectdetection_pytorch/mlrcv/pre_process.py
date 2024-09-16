import numpy as np
import torch
from typing import Optional

def heatmap_object(img: np.ndarray, bounding_box: dict, heatmap: np.ndarray) -> np.ndarray:
    """
    This function generates the heatmaps over the objects given the input image:

    Args:
        - img (np.ndarray): input image
        - bounding_box (dict): labels with one object bounding box
        - heatmap (np.ndarray): heatmap of the current input img

    Returns:
        - heatmap (np.ndarray): output heatmap with the current object heatmap added
    """
    img = np.asarray(img).astype(np.float)

    center = bounding_box['center']
    bndbox = bounding_box['bndbox']

    x1, y1, x2, y2 = bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']
    x0, y0 = center

    #gaussian parameters
    sigma = 3
    A = 1

    x1, y1, x2, y2 = int(x1)//4, int(y1)//4, int(x2)//4, int(y2)//4

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(heatmap.shape[1], x2), min(heatmap.shape[0], y2)


    for i in range(x1, x2):
        for j in range(y1, y2):
            x = ((i - x0) ** 2) / (2 * sigma ** 2)
            y = ((j - y0) ** 2) / (2 * sigma ** 2)
            heatmap[j, i] = max(heatmap[j, i], A * np.exp(-(x + y)))

    return heatmap

def sizemap_object(img: np.ndarray, bounding_box: dict, sizemap: np.ndarray) -> np.ndarray:
    """
    This function generates the sizemaps over the objects given the input image:

    Args:
        - img (np.ndarray): input image
        - bounding_box (dict): labels with one object bounding box
        - sizemap (np.ndarray): sizemap of the current input img

    Returns:
        - sizemap (np.ndarray): output sizemap with the current object sizemap added
    """
    bounding_box = bounding_box['bndbox']
    center = bounding_box['center']
    
    x0 = center[0]
    y0 = center[1]
    xmax = bounding_box['xmax']
    xmin = bounding_box['xmin']
    ymax = bounding_box['ymax']
    ymin = bounding_box['ymin']

    height = ymax - ymin
    width = xmax - xmin


    sizemap[x0, y0, 0] = height
    sizemap[x0, y0, 1] = width

    return sizemap