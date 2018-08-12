import click
import cv2
import onegan
import torch
import torchvision.transforms as T
from PIL import Image

import numpy as np
import tensorflow as tf

import onnx
from onnx_tf.backend import prepare

import scipy

torch.backends.cudnn.benchmark = True


class Predictor:

    def __init__(self, input_size):
        self.model = self.build_model()
        self.colorizer = onegan.extension.Colorizer(
            colors=[
                [249, 69, 93], [255, 229, 170], [144, 206, 181],
                [81, 81, 119], [241, 247, 210]])
        self.transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

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
        model = onnx.load('models/onnx/my_model.onnx')
        tf_rep = prepare(model)
        return tf_rep

    def process(self, raw):

        def _batched_process(batched_img):
            print(batched_img.shape)

            score = self.model.run(batched_img)
            #score, _ = self.sess.run(logits, feed_dict={images : rand})
            print(score._0.shape)
            score = torch.tensor(score._0)
            print(score.shape)
            
            _, output = torch.max(score, 1)

            image = (batched_img / 2 + .5)
            layout = self.colorizer.apply(output.data.cpu())
            return image * .6 + layout * .4

        img = raw
        batched_img = self.transform(img).unsqueeze(0)
        canvas = _batched_process(batched_img)
        result = canvas.squeeze().permute(1, 2, 0).numpy()
        raw = np.array(raw)
        return cv2.resize(result, (raw.shape[1], raw.shape[0]))


@click.command()
@click.option('--input_size', default=(320, 320), type=(int, int))
def main(input_size):

    demo = Predictor(input_size)

    img = Image.open('/app/data/lsun_room/images/0a5aa9806899b1e8055ee917381bb185d8f34d08.jpg').resize(input_size)

    output = demo.process(img)

    scipy.misc.imsave('output/super_res_output.jpg', output[:, :, ::-1])


if __name__ == '__main__':
    main()