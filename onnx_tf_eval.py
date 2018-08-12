import click
import cv2
from PIL import Image

import numpy as np
import tensorflow as tf

import onnx
from onnx_tf.backend import prepare

import scipy

# https://github.com/leVirve/lsun-room/issues/1

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
                x:input_images # < 45
            })
            # I taught a neural net to recognise when a sum of numbers is bigger than 45
            # it should return False in this case
        #print(y_out) # [[ False ]] Yay, it works!
        print(y_out.shape)
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
            return self.build_model_from_onnx(filename)
        elif '.pb' in filename:
            return Model(filename)
        raise Exception('Format not supported.')

    def build_model_from_onnx(self, filename):
        model = onnx.load(filename)
        tf_rep = prepare(model)
        print(tf_rep.predict_net.tensor_dict[tf_rep.predict_net.external_input[0]].name)
        print(tf_rep.predict_net.tensor_dict[tf_rep.predict_net.external_output[0]].name)
        return tf_rep

    def process(self, raw):

        def _batched_process(batched_img):
            print(batched_img.shape)

            score = self.model.run(batched_img)
            score = score
            
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
@click.option('--model_path', default=['models/tf/my_model.pb', 'models/onnx/my_model.onnx'][0], type=str)
def main(input_size, model_path):

    demo = Predictor(model_path, input_size)

    img = Image.open('/app/data/lsun_room/images/00fa6667796186b94f6efae1b24fd6933ad96843.jpg').resize(input_size)

    output = demo.process(img)

    scipy.misc.imsave('output/super_res_output.jpg', output)


if __name__ == '__main__':
    main()