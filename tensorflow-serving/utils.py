import cv2
from PIL import Image

import numpy as np
import tensorflow as tf

import enum

import matplotlib.pyplot as plt
from sklearn import svm

import scipy

class Layout(enum.Enum):
    frontal = 0
    left = 1
    right = 2
    floor = 3
    ceiling = 4
    
class Colorizer():
    def __init__(self, colors, num_output_channel=3):
        self.colors = self.normalized_color(colors)
        self.num_label = len(colors)
        self.num_channel = num_output_channel

    @staticmethod
    def normalized_color(colors):
        colors = np.array(colors, 'float32')
        if colors.max() > 1:
            colors = colors / 255
        return colors

    def apply(self, label):
        if label.ndim == 2:
            label = label[np.newaxis, :]
        if label.ndim == 4:
            label = label.squeeze(1)
        assert label.ndim == 3, label.ndim
        n, h, w = label.shape

        canvas = np.zeros((n, h, w, self.num_channel))
        for lbl_id in range(self.num_label):
            if canvas[label == lbl_id].shape[0]:
                canvas[label == lbl_id] = self.colors[lbl_id]

        return canvas.transpose((0, 3, 1, 2))

class Model:
    def __init__(self, frozen_graph_filename):
        """
        https://www.tensorflow.org/mobile/prepare_models
        https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
        """
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")

        self.graph = graph
        #tf.train.write_graph(graph_def, 'pbtxt/', 'protobuf.pbtxt', as_text=True)

    
    def run(self, input_images):
       
        # We access the input and output nodes 
        x = self.graph.get_tensor_by_name('prefix/0:0')
        y = self.graph.get_tensor_by_name('prefix/concat_111:0')

        # We launch a Session
        with tf.Session(graph=self.graph) as sess:
            # Note: we don't nee to initialize/restore anything
            # There is no Variables in this graph, only hardcoded constants 
            y_out = sess.run(y, feed_dict={
                x:input_images 
            })
        return y_out

class Predictor:

    def __init__(self, model_path, input_size):
        self.colorizer = Colorizer(
            colors=[
                [249, 69, 93], [255, 229, 170], [144, 206, 181],
                [81, 81, 119], [241, 247, 210]])
        self.model = self.build_model(model_path)

    def build_model(self, filename):
        if '.onnx' in filename:
            raise Exception('Use appropriate ONNX predictor, or convert to TF pb file.')
        elif '.pb' in filename:
            return Model(filename)
        raise Exception('Format not supported.')

    def process(self, raw):

        def _batched_process(batched_img):
            print(batched_img.shape)

            score = self.model.run(batched_img)
            score = score
            
            output = np.argmax(score, 1)

            image = (batched_img / 2 + .5)
            layout = self.colorizer.apply(output)
            return image *.0 + layout * 1, output

        raw = np.array(raw)
        raw = cv2.normalize(raw.astype(np.float32), None, alpha=-0.5, beta=0.5,\
                                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        batched_img = np.expand_dims(raw, axis=0).transpose((0, 3, 1, 2))
        canvas, label_map = _batched_process(batched_img)
        result = canvas.squeeze().transpose((1, 2, 0))
        resize = lambda x: cv2.resize(x, (raw.shape[1], raw.shape[0]))
        return resize(result), resize(label_map.squeeze(0))


def get_data(label_map, label_1, label_2):
    labels_coord_1 = np.array(np.where(label_map == label_1))
    labels_coord_2 = np.array(np.where(label_map == label_2))
    labels_coord = np.concatenate((labels_coord_1, labels_coord_2), axis=1).transpose(1,0)
    labels_coord = np.flip(labels_coord, axis=1)
    labels_value_1 = np.ones(labels_coord_1.shape[1]) * label_1
    labels_value_2 = np.ones(labels_coord_2.shape[1]) * label_2
    labels_value = np.concatenate((labels_value_1, labels_value_2))
    return labels_coord, labels_value

def gen_edge_map(layout):
    import cv2
    lbl = cv2.GaussianBlur(layout.astype('uint8'), (3, 3), 0)
    edge = cv2.Laplacian(lbl, cv2.CV_64F)
    activation = cv2.dilate(np.abs(edge), np.ones((5, 5), np.uint8), iterations=1)
    activation[activation != 0] = 1
    return cv2.GaussianBlur(activation, (15, 15), 5)

def connectedness(label_map, label_1, label_2):
    label1_points = np.array(np.where(label_map == label_1)).T
    label2_points = np.array(np.where(label_map == label_2)).T
    for pt in label1_points:
        for i in range(-8,8):
            for j in range(-8,8):
                neigh = pt + np.array([i, j])
                if np.array(np.where(label2_points == neigh)).any():
                    return True
    return False

def discard_smallest_blobs(array, label_num, discard_value=-1, plot_diff=False):
    """
    Discard the largest area of a specific value in a 2D numpy array
    https://stackoverflow.com/questions/20110232/python-efficient-way-to-find-the-largest-area-of-a-specific-value-in-a-2d-nump
    """
    from scipy import ndimage

    colorizer = Colorizer(
                colors=[
                    [249, 69, 93], [255, 229, 170], [144, 206, 181],
                    [81, 81, 119], [241, 247, 210]])

    before_img = colorizer.apply(array).squeeze().transpose((1, 2, 0))

    label, num_label = ndimage.label(array == label_num)
    size = np.bincount(label.ravel())
    biggest_label = size[1:].argmax() + 1
    clump_mask = label == biggest_label
    array[(array == label_num) & (clump_mask == False)] = discard_value

    after_img = colorizer.apply(array).squeeze().transpose((1, 2, 0))

    if plot_diff:
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(before_img, cmap = 'gray', interpolation = 'bicubic')
        axarr[1].imshow(after_img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    return array

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)
