import click
import cv2
from PIL import Image

import numpy as np
import tensorflow as tf

import enum

import matplotlib.pyplot as plt
from sklearn import svm

import scipy

from utils import (Predictor, Layout, connectedness, get_data, gen_edge_map)

def find_best_separation(label_map, label_1, label_2):
    X, y = get_data(label_map, label_1, label_2) 

    # fit the model and get the separating hyperplane
    clf = svm.LinearSVC(C=500.0, dual=False)
    clf.fit(X, y)

    W = clf.coef_[0]
    a = - W[0] / W[1]
    b = - (clf.intercept_[0]) / W[1]

    xx = np.linspace(0, 320)
    yy = a * xx + b

    pt1 = int(xx[0]), int(yy[0])
    pt2 = int(xx[-1]), int(yy[-1])
    return pt1, pt2

def gen_linear_layout_map(label_map, output):
    set_label = [label for label in Layout if label.value in np.unique(label_map)]
    edge = gen_edge_map(label_map)
    label_map[edge == 0] = -1 
    labels = {
        Layout.frontal: (Layout.left, Layout.right, Layout.floor, Layout.ceiling),
        Layout.left: (Layout.right, Layout.floor, Layout.ceiling),
        Layout.right: (Layout.floor, Layout.ceiling),
        }
    edge = gen_edge_map(label_map)
    label_map[edge == 0] = -1 
    lines = {x:{} for x in set_label}
    for label_1, label_1_labels in labels.items():
        if label_1 not in set_label:
            continue
        for label_2 in label_1_labels:
            if label_2 not in set_label:
                continue
            if not connectedness(label_map, label_1.value, label_2.value):
                continue
            
            pt1, pt2 = find_best_separation(label_map, label_1.value, label_2.value)
            cv2.line(output, pt1, pt2, [0,0,1], thickness=1)
            print(lines)
            lines[label_1][label_2] = (pt1, pt2)

    poly_corners = {x:[] for x in set_label}
    for label_1, label_1_dict in lines.items():
        #for label_2, label_1_label_2_lines in label_2_dict.items():
        if label_1 == Layout.frontal:
            pt1, pt2 = label_1_dict[Layout.left]
            pt3, pt4 = label_1_dict[Layout.ceiling]
            corner = get_intersect(pt1, pt2, pt3, pt4)
            poly_corners[label_1].append(corner)

            pt1, pt2 = label_1_dict[Layout.left]
            pt3, pt4 = label_1_dict[Layout.floor]
            corner = get_intersect(pt1, pt2, pt3, pt4)
            poly_corners[label_1].append(corner)

            pt1, pt2 = label_1_dict[Layout.right]
            pt3, pt4 = label_1_dict[Layout.floor]
            corner = get_intersect(pt1, pt2, pt3, pt4)
            poly_corners[label_1].append(corner)

            pt1, pt2 = label_1_dict[Layout.right]
            pt3, pt4 = label_1_dict[Layout.ceiling]
            corner = get_intersect(pt1, pt2, pt3, pt4)
            poly_corners[label_1].append(corner)

        print(poly_corners)
        
    return lines, output


@click.command()
@click.option('--input_size', default=(320, 320), type=(int, int))
@click.option('--model_path', default=['models/tf/my_model.pb', 'models/onnx/my_model.onnx'][0], type=str)
def main(input_size, model_path):
    demo = Predictor(model_path, input_size)

    img = Image.open('/app/hubstairs.jpg').resize(input_size)

    output, label_map = demo.process(img)

    lines, output = gen_linear_layout_map(label_map, output)

    minmax = lambda x, y: (min(320, max(0, x)), min(320, max(0, y)))
    for pt1, pt2 in lines:
        cv2.circle(output, minmax(*pt1), 10, [255,0,0], thickness=1, lineType=8, shift=0) 
        cv2.circle(output, minmax(*pt2), 10, [255,0,0], thickness=1, lineType=8, shift=0) 
        for pt3, pt4 in lines:
            if pt1 != pt3 and pt2 != pt4:
                x, y = get_intersect(pt1, pt2, pt3, pt4)
                cv2.circle(output, (int(x), int(y)), 5, [255,0,0], thickness=1, lineType=8, shift=0) 
                plt.imshow(output, cmap = 'gray', interpolation = 'bicubic')
                plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                plt.show()                

    scipy.misc.imsave('output/super_res_output.jpg', output)


if __name__ == '__main__':
    main()