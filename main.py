import importlib
import logging

import torch
import onegan

from torch.autograd import Variable

from trainer import core
from trainer.model import ResPlanarSeg

import scipy.misc

import onnx
import onnx_tf
import onnx_tf.backend as tf_backend
import tensorflow as tf


def create_dataset(args):
    assert args.batch_size > 1

    module = importlib.import_module(f'datasets.{args.dataset}')
    Dataset = getattr(module, {
        'sunrgbd': 'SunRGBDDataset',
        'lsunroom': 'LsunRoomDataset',
        'hedau': 'HedauDataset',
    }[args.dataset])
    args.num_class = Dataset.num_classes
    kwargs = {'collate_fn': onegan.io.universal_collate_fn}

    return (Dataset(phase, args=args).to_loader(**kwargs)
            for phase in ['train', 'val'])


def create_model(args):
    return {
        'resnet': lambda: ResPlanarSeg(num_classes=args.num_class, pretrained=True, base='resnet101')
    }[args.arch]()


def create_optim(args, model, optim='sgd'):
    return {
        'adam': lambda: torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999)),
        'sgd': lambda: torch.optim.SGD(model.parameters(), lr=args.lr / 10, momentum=0.9)
    }[optim]()


def hyperparams_search(args):
    search_hyperparams = {
        'arch': ['vgg', 'mike'],
        'image_size': [320],
        'edge_factor': [0, 0.2, 0.4],
    }

    import itertools
    for i, params in enumerate(itertools.product(*search_hyperparams.values())):
        for key, val in zip(search_hyperparams.keys(), params):
            args[key] = val
        args.name = '{}{}_e{}_g{}'.format(*params)
        print(f'Experiment#{i + 1}:', args.name)
        main(args)

def onnx2tf(onnx_model_fn, tf_model_fn):
    """
    https://github.com/onnx/onnx-tensorflow/issues/231
    https://github.com/onnx/onnx-tensorflow/blob/master/doc/API.md
    """
    model = onnx.load(onnx_model_fn)
    tf_rep = tf_backend.prepare(model)
    tf_rep.export_graph(path=tf_model_fn)

def main(args):
    log = logging.getLogger('room')
    log.info(''.join([f'\n-- {k}: {v}' for k, v in args.items()]))

    train_loader, val_loader = create_dataset(args)
    model = create_model(args)

    if args.gpu:
        get_model = lambda: model.cuda()
    else:
        get_model = lambda: model.cpu()
    

    if args.phase == 'train':
        training_estimator = core.training_estimator(
            torch.nn.DataParallel(get_model()),
            create_optim(args, model, optim=args.optim), args)
        training_estimator(train_loader, val_loader, epochs=args.epoch)

    if args.phase in ['eval', 'eval_search']:
        core_fn = core.evaluation_estimator if args.phase == 'eval' else core.weights_estimator
        evaluate_estimator = core_fn(torch.nn.DataParallel(get_model()), args)
        evaluate_estimator(val_loader)
    
    if args.phase == 'export':
        dummy_input = Variable(torch.randn(1, 3, 320, 320)).cuda()
        checkpoint = onegan.extension.Checkpoint(name=args.name, save_interval=5)
        model = checkpoint.load(args.pretrain_path, model=get_model(), remove_module=True)
        torch.onnx.export(model, dummy_input, "my_model.onnx", export_params=True)
        print("Exported to ONNX format.")
        onnx_model_fn = 'my_model.onnx'
        tf_model_fn = 'my_model.pbtxt'
        model = onnx.load(onnx_model_fn)
        tf_rep = tf_backend.prepare(model)
        tf_rep.export_graph(tf_model_fn)
        print("Exported to TF format.")

if __name__ == '__main__':
    parser = onegan.option.Parser(description='Indoor room corner detection', config='./config.yml')
    parser.add_argument('--name', help='experiment name')
    parser.add_argument('--folder', help='where\'s the dataset')
    parser.add_argument('--dataset', default='lsunroom', choices=['lsunroom', 'hedau', 'sunrgbd'])
    parser.add_argument('--phase', default='eval', choices=['train', 'eval', 'eval_search', 'export', 'tf_eval'])
    parser.add_argument('--gpu', default=False, type=bool, help='use gpu')

    # data
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--use_edge', action='store_true')
    parser.add_argument('--use_corner', action='store_true')
    parser.add_argument('--datafold', type=int, default=1)

    # outout
    parser.add_argument('--tri_visual', action='store_true')

    # network
    parser.add_argument('--arch', default='resnet')
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--disjoint_class', action='store_true')
    parser.add_argument('--pretrain_path', default='')

    # hyper-parameters
    parser.add_argument('--l1_factor', type=float, default=0.0)
    parser.add_argument('--l2_factor', type=float, default=0.0)
    parser.add_argument('--edge_factor', type=float, default=0.0)
    parser.add_argument('--focal_gamma', type=float, default=0)
    args = parser.parse()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.StreamHandler(), ])
    main(args)
