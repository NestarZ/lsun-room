# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import inspect
import logging
from enum import Enum
from collections import defaultdict

import tqdm
import torch

import onegan
import onegan.loss as losses
from onegan.option import AttrDict
from onegan.extension import History, TensorBoardLogger, GANCheckpoint


class Events(Enum):
    """ Events for the estimator  """
    STARTED = 'started'
    END = 'end'
    ITERATION_START = 'iteration_start'
    ITERATION_END = 'iteration_end'
    EPOCH_START = 'epoch_start'
    EPOCH_END = 'epoch_end'


class EstimatorEventMixin:
    """ Mixin for the event-triggered estimator.

    Maim implementation comes from `https://github.com/pytorch/ignite/blob/master/ignite/engine/engine.py`
    """

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        """ Add an event handler to be executed when the specified event is triggered.

        Args:
            event_name (Events): event the handler attach to
            handler (Callable): the callable function that should be invoked
            *args: optional args to be passed to `handler`
            **kwargs: optional keyword args to be passed to `handler`

        Notes:
            The handler function's first argument will be `self` (the `Estimator`).

        Examples:

            >>> def print_epoch(estimator):
            >>>    print("Epoch: {}".format(estimator.state.epoch))
            >>> estimator.add_event_handler(Events.EPOCH_END, print_epoch)
        """
        if event_name not in Events:
            self._log.error(f'attempt to add event handler to an invalid event {event_name}')
            raise ValueError(f'Event {event_name} is not a valid event')

        self._check_signature(handler, 'handler', *args, **kwargs)
        self._events[event_name].append((handler, args, kwargs))
        self._log.debug(f'Handler added for event {event_name}')

    def on(self, event_name, *args, **kwargs):
        """ Decorator shortcut for add_event_handler.

        Args:
            event_name (Events): event the handler attach to
            *args: optional args to be passed to `handler`
            **kwargs: optional keyword args to be passed to `handler`
        """
        def decorator(f):
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f
        return decorator

    def _check_signature(self, fn, fn_description, *args, **kwargs):
        exception_msg = None

        signature = inspect.signature(fn)
        try:
            signature.bind(self, *args, **kwargs)
        except TypeError as exc:
            fn_params = list(signature.parameters)
            exception_msg = str(exc)

        if exception_msg:
            passed_params = [self] + list(args) + list(kwargs)
            raise ValueError(f'Error adding {fn} "{fn_description}": '
                             f'takes parameters {fn_params} but will be called with {passed_params} '
                             f'({exception_msg})')

    def _trigger(self, event_name, *args):
        self._log.debug(f'trigger handlers for event {event_name}')
        for handle in self._events[event_name]:
            evt_handler, evt_args, evt_kwargs = handle
            evt_handler(self, *(args + evt_args), **evt_kwargs)


class Estimator:
    """ Base estimator for functional support. """

    def load_checkpoint(self, weight_path, remove_module=False, resume=False) -> None:
        """ load checkpoint if internal ``saver`` is not `None`. """
        if not hasattr(self, 'saver') or self.saver is None:
            return
        self.saver.load(weight_path, self.model, remove_module=remove_module, resume=resume)

    def save_checkpoint(self, save_optim=False) -> None:
        """ save checkpoint if internal ``saver`` is not `None`. """
        if not hasattr(self, 'saver') or self.saver is None:
            return
        optim = self.optimizer if save_optim else None
        self.saver.save(self.model, optim, self.state.epoch + 1)

    def adjust_learning_rate(self, monitor_val) -> None:
        """ adjust the learning rate if internal ``lr_scheduler`` is not `None`. """
        if not hasattr(self, 'lr_scheduler') or self.lr_scheduler is None:
            return
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(monitor_val)
        else:
            self.lr_scheduler.step()


'''
Event-trigger estimator
'''


def epoch_end_logging(estmt):
    estmt.tensorboard_epoch_logging(scalar=estmt.history.metric)


def iteration_end_logging(estmt):
    summary = estmt.state.get('summary', {})
    prefix, image, histogram = summary.get('prefix'), summary.get('image'), summary.get('histogram')
    estmt.tensorboard_logging(image=image, prefix=prefix)
    estmt.tensorboard_logging(histogram=histogram, prefix=prefix)


def adjust_learning_rate(estmt):
    estmt.adjust_learning_rate(estmt.history.get('loss/loss_val'))


def save_checkpoint(estmt):
    estmt.save_checkpoint()


class OneEstimator(EstimatorEventMixin, Estimator):
    r""" Estimator for network training and evaluation.

    Args:
        model (torch.nn.Module): defined model for estimator.
        optimizer (torch.optim, optional): optimizer for model training.
        lr_scheduler (torch.optim.lr_scheduler, optional): learning rate scheduler for
            model training.
        logger (extension.TensorBoardLogger, optional): training state logger (default: None).
        saver (extension.Checkpoint, optional): checkpoint persistence (default: None).
        default_handlers (bool): turn on/off the defalt handlers (default: False).

    Attributes:
        history (extension.History): internal statistics of training state.
    """

    def __init__(self, model, optimizer=None, lr_scheduler=None, logger=None, saver=None, default_handlers=False):
        self.model = model

        # can leave empty
        self.optimizer = optimizer

        # optional
        self.lr_scheduler = lr_scheduler
        self.saver = saver
        self.logger = logger

        # internal
        self.history = History()
        self.state = AttrDict(epoch=0)
        self._events = defaultdict(list)
        self._hist_dict = defaultdict(list)
        self._log = logging.getLogger('onegan.OneEstimator')
        #self._log.setLevel(level=logging.DEBUG)

        if default_handlers:
            self.add_default_event_handlers()
        self._log.info(f'OneEstimator is initialized')

    def add_default_event_handlers(self):
        #self.add_event_handler(Events.ITERATION_END, iteration_end_logging)
        self.add_event_handler(Events.EPOCH_END, epoch_end_logging)
        self.add_event_handler(Events.EPOCH_END, save_checkpoint)
        self.add_event_handler(Events.EPOCH_END, adjust_learning_rate)

    def tensorboard_logging(self, image=None, histogram=None, prefix=None):
        ''' wrapper in estimator for Tensorboard logger.

        Args:
            image: dict() of a list of images
            histogram: dict() of tensors for accumulated histogram
            prefix: prefix string for keyword-image
        '''
        if not hasattr(self, 'logger') or self.logger is None:
            return

        if image and prefix:
            self.logger.image(image, self.state.epoch, prefix)
            self._log.debug('tensorboard_logging logs images')

        if histogram and prefix:
            for tag, tensor in histogram.items():
                self._hist_dict[f'{prefix}{tag}'].append(tensor.clone())
            self._log.debug('tensorboard_logging accumulate histograms')

    def tensorboard_epoch_logging(self, scalar=None):
        ''' wrapper in estimator for Tensorboard logger.

        Args:
            scalar: dict() of a list of scalars
        '''
        if not hasattr(self, 'logger') or self.logger is None:
            return

        self.logger.scalar(scalar, self.state.epoch)
        self._log.debug('tensorboard_epoch_logging logs scalars')

        if self._hist_dict:
            kw_histograms = {tag: torch.cat(tensors) for tag, tensors in self._hist_dict.items()}
            self.logger.histogram(kw_histograms, self.state.epoch)
            self._hist_dict = defaultdict(list)
            self._log.debug('tensorboard_epoch_logging logs histograms')

    def run(self, train_loader, validate_loader, closure_fn, epochs, longtime_pbar=False):
        epoch_range = tqdm.trange(epochs, desc='Training Procedure') if longtime_pbar else range(epochs)

        for epoch in epoch_range:
            self.history.clear()
            self.state.epoch = epoch
            self._trigger(Events.EPOCH_START)

            self.train(train_loader, closure_fn, longtime_pbar)
            self.evaluate(validate_loader, closure_fn, longtime_pbar)
            
            self._trigger(Events.EPOCH_END)
            self._log.debug(f'OneEstimator epoch#{epoch} end')

    def train(self, data_loader, update_fn, longtime_pbar=False):
        self.model.train()
        progress = tqdm.tqdm(data_loader, desc=f'Epoch#{self.state.epoch + 1}', leave=not longtime_pbar)

        for data in progress:
            self._trigger(Events.ITERATION_START)

            loss, accuracy = update_fn(self.model, data)
            progress.set_postfix(self.history.add({**loss, **accuracy}))
            self.optimizer.zero_grad()
            loss['loss/loss'].backward()
            self.optimizer.step()

            self.state.update({**loss, **accuracy})

            self._trigger(Events.ITERATION_END)

    def evaluate(self, data_loader, inference_fn, longtime_pbar=False):
        self.model.eval()
        progress = tqdm.tqdm(data_loader, desc='evaluating', leave=not longtime_pbar)

        with torch.no_grad():
            for data in progress:
                self._trigger(Events.ITERATION_START)

                log_values = inference_fn(self.model, data)
                loss, accuracy = log_values if isinstance(log_values, tuple) else (log_values, {})
                scalars_dict = {**loss, **accuracy}
                progress.set_postfix(self.history.add(scalars_dict, log_suffix='_val'))
                
                self.state.update(scalars_dict)
                
                self._trigger(Events.ITERATION_END)

                