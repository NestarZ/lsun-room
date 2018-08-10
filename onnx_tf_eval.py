import click
import cv2
from PIL import Image

import numpy as np
import tensorflow as tf

import onnx
from onnx_tf.backend import prepare

import scipy

class Colorizer():
    def __init__(self, colors, num_output_channel=3):
        self.colors = self.normalized_color(colors)
        self.num_label = len(colors)
        self.num_channel = num_output_channel
        # self.transform = T.Compose([
        #     T.Resize(input_size),
        #     T.ToTensor(),
        #     T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        # ])

    @staticmethod
    def normalized_color(colors):
        colors = np.array(colors, 'float32')
        if colors.max() > 1:
            colors = colors / 255
        return colors

    def apply(self, label):
        if label.ndim == 4:
            label = label.squeeze(1)
        assert label.ndim == 3, label.ndim
        n, h, w = label.shape

        canvas = np.zeros((n, h, w, self.num_channel))
        input(label)
        for lbl_id in range(self.num_label):
            if canvas[label == lbl_id].shape[0]:
                canvas[label == lbl_id] = self.colors[lbl_id]

        return canvas.transpose((0, 3, 1, 2))

class Predictor:

    def __init__(self, input_size):
        self.colorizer = Colorizer(
            colors=[
                [249, 69, 93], [255, 229, 170], [144, 206, 181],
                [81, 81, 119], [241, 247, 210]])
        self.model = self.build_model()

    def build_model(self):
        return self.build_model_from_onnx()

        # filename = "./models/saved_model.pb"
        # # We load the protobuf file from the disk and parse it to retrieve the
        # # unserialized graph_def
        # with tf.Session() as sess:
        #     with tf.gfile.GFile(filename, "rb") as f:
        #         graph_def = tf.GraphDef()
        #         graph_def.ParseFromString(f.read())
        #           with tf.Graph().as_default() as graph:
        #             tf.import_graph_def(graph_def, name="")
        # #train_writer = tf.summary.FileWriter('./log/')
        # #train_writer.add_graph(sess.graph)
        # return graph, sess

    def build_model_from_onnx(self):
        model = onnx.load('my_model.onnx')
        tf_rep = prepare(model)
        return tf_rep

    def process(self, raw):

        def _batched_process(batched_img):
            print(batched_img.shape)

            score = self.model.run(batched_img)
            #score, _ = self.sess.run(logits, feed_dict={images : rand})
            score = score._0
            
            output = np.argmax(score, 1)

            image = (batched_img / 2 + .5)
            layout = self.colorizer.apply(output)
            return image *.3 + layout * 1
        raw = np.array(raw)
        raw = cv2.normalize(raw.astype(np.float32), None, alpha=-0.5, beta=0.5,\
                                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        batched_img = np.expand_dims(raw, axis=0).transpose((0, 3, 1, 2))
        canvas = _batched_process(batched_img)
        result = canvas.squeeze().transpose((1, 2, 0))
        return cv2.resize(result, (raw.shape[1], raw.shape[0]))


@click.command()
@click.option('--input_size', default=(320, 320), type=(int, int))
def main(input_size):

    demo = Predictor(input_size)

    img = Image.open('/app/data/lsun_room/images/0a5aa9806899b1e8055ee917381bb185d8f34d08.jpg').resize(input_size)

    output = demo.process(img)

    scipy.misc.imsave('output/super_res_output.jpg', output)


if __name__ == '__main__':
    main()