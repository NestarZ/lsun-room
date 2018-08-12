import click
import cv2
from PIL import Image

import numpy as np
import tensorflow as tf

import enum

import matplotlib.pyplot as plt

import scipy
from scipy.spatial import ConvexHull

from utils import (Predictor, Layout, discard_smallest_blobs)

import os, random

from glob import glob

"""
Following:
https://pdfs.semanticscholar.org/7024/a92186b81e6133dc779f497d06877b48d82b.pdf?_ga=2.54181869.497995160.1510977308-665742395.1510465328
http://www.cs.toronto.edu/~fidler/slides/CVPR15tutorial/room_layout.pdf
https://stackoverflow.com/questions/38409156/minimal-enclosing-parallelogram-in-python
https://math.stackexchange.com/questions/871867/rotation-matrix-of-triangle-in-3d
https://math.stackexchange.com/questions/62936/transforming-2d-outline-into-3d-plane/63100#63100
"""
COLORS = [[249, 69, 93], [255, 229, 170], [144, 206, 181],
            [81, 81, 119], [241, 247, 210]]

@click.command()
@click.option('--input_size', default=(320, 320), type=(int, int))
@click.option('--model_path', default=['models/tf/my_model.pb', 'models/onnx/my_model.onnx'][0], type=str)
def main(input_size, model_path):
    demo = Predictor(model_path, input_size)

    while True:
        glob_pattern = os.path.join("/app/data/lsun_room/images/", '*.jpg')
        img_fn = random.choice(glob(glob_pattern)) 

        img = np.array(Image.open(img_fn).resize(input_size))

        output, label_map = demo.process(img)
        output_layout = output.copy()
        output_layout_edges = output.copy()
        
        for label in np.unique(label_map):
            label_map = discard_smallest_blobs(label_map, label, discard_value=-1, plot_diff=False)
            pts = np.flip(np.array(np.where(label_map == label)).T, axis=1)
            hull = ConvexHull(pts, qhull_options='Qt')
            #rot_angle, area, width, height, center_point, corner_points = minBoundingRect(pts[hull.vertices])
            corner_points = pts[hull.vertices]
            #corner_points = minimum_bounding_rectangle(pts)
            corner_points = corner_points.astype(np.int32)
            polygon_pts = corner_points
            polygon_pts_rdp = rdp(corner_points, epsilon=10)
            cv2.fillPoly(output_layout, pts=[polygon_pts], color=COLORS[label])
            cv2.polylines(output_layout_edges, [polygon_pts], True,COLORS[label])

        f, axarr = plt.subplots(1,4)
        axarr[0].imshow(img, interpolation = 'bicubic')
        axarr[1].imshow(output.astype('float32'), interpolation = 'bicubic')
        axarr[2].imshow(output_layout.astype('uint8'), interpolation = 'bicubic')
        axarr[3].imshow(output_layout_edges.astype('uint8'), interpolation = 'bicubic')
        plt.title(img_fn)
        plt.show()

        alpha = 0.9
        output = cv2.addWeighted(output, alpha, output_layout, 1-alpha, 0)

        scipy.misc.imsave('output/super_res_output.jpg', output)


if __name__ == '__main__':
    main()