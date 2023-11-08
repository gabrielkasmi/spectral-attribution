# -*- coding: utf-8 -*-

import os
import numpy as np
from shapely.geometry import Polygon, Point

def generate_mask_from_bbox(name, bbox_dir, image_size):
    """
    generates a binary mask from the xml file

    return a mask, which is the union of all bboxes
    
    """


    # open the xml file and get the bounding boxes
    bbox_path = os.path.join(bbox_dir, name + '.xml')
    tree = ET.parse(bbox_path)
    root = tree.getroot()
    sample_annotations = []

    for neighbor in root.iter('bndbox'):
        xmin = int(neighbor.find('xmin').text)
        ymin = int(neighbor.find('ymin').text)
        xmax = int(neighbor.find('xmax').text)
        ymax = int(neighbor.find('ymax').text)
        
        sample_annotations.append([xmin, ymin, xmax, ymax])

    # convert the sample annotations into polygons

    polygons = []
    for sa in sample_annotations:

        xmin, ymin, xmax, ymax = sa

        bbox_coords = [
            [xmin, ymin],
            [xmin, ymax],
            [xmax, ymax],
            [xmax, ymin],
        ]

    bbox_coords = np.array(bbox_coords)
    polygons.append(Polygon(bbox_coords))

    # generate a mask
    mask = np.zeros(image_size).astype(np.uint8)

    for ix, iy in np.ndindex(mask.shape):

        p = Point(ix,iy)

        for polygon in polygons:
            if polygon.contains(p):

                mask[ix, iy] = 255

        
    return mask.transpose()




