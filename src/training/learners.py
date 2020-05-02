import torch
import torch.nn.functional as F
from utils import crop_range
from metrics import match_up
from flows import MetricFlow, ModuleFlow


class Learner:

    def __init__(
        self,
        model: ModuleFlow,
        meter: MetricFlow,
        optim,
        grad_accumulation=1
    ):
        self.model = model
        self.meter = meter
        self.optim = optim
        self.step = 0
        self.model.zero_grad()
        self.grad_accumulation = grad_accumulation
        assert isinstance(self.grad_accumulation, int)
        assert self.grad_accumulation >= 1

    def _model_run(self, data, training=True):
        if training:
            with torch.set_grad_enabled(True):
                self.model.train()
                return self.model(data)
        else:
            with torch.set_grad_enabled(False):
                self.model.eval()
                return self.model(data)

    def _evaluate(self, data, training=True):
        with torch.set_grad_enabled(training):
            return self.meter(data)

    def _backpropagation(self, loss):
        if self.grad_accumulation >= 1:
            loss = loss / self.grad_accumulation
        loss.backward()
        self.step += 1

        if self.step % self.grad_accumulation == 0:
            self.optim.step()
            self.model.zero_grad()

    def learn(self, data):
        outputs = self._model_run(data, training=True)
        data.update(outputs)
        results = self._evaluate(data, training=True)
        self._backpropagation(results['loss'])
        return results

    def infer(self, data):
        outputs = self._model_run(data, training=False)
        data.update(outputs)
        return self._evaluate(data, training=False)


class DisLearner(Learner):

    def __init__(self, model: ModuleFlow, meter: MetricFlow, optim, **kwargs):
        super().__init__(model, meter, optim, **kwargs)

    def learn(self, data):
        assert 'prediction' in data
        assert 'label' in data

        label_mask = (data['label'] >= 0).unsqueeze(1)
        if not torch.any(label_mask):
            return {
                'DIS_TRUTH': 0.,
                'DIS_FAKE': 0.,
            }

        # train on ground truth
        n_classes = data['prediction'].shape[1]
        onehot_label = F.one_hot(data['label'], n_classes)
        onehot_label = onehot_label.permute((0, 4, 1, 2, 3))
        onehot_label = onehot_label.float()
        outputs = self._model_run({'label': onehot_label}, training=True)
        results = self._evaluate(
            {
                'confidence_map': outputs['confidence_map'][label_mask],
                'truth': True
            },
            training=True
        )
        truth_loss = results['loss']
        self._backpropagation(truth_loss)

        # train on model prediction
        model_prediction = F.softmax(data['prediction'].detach(), dim=1)
        outputs = self._model_run({'label': model_prediction}, training=True)
        results = self._evaluate(
            {
                'confidence_map': outputs['confidence_map'][label_mask],
                'truth': False
            },
            training=True
        )
        fake_loss = results['loss']
        self._backpropagation(fake_loss)

        return {
            'DIS_TRUTH': truth_loss,
            'DIS_FAKE': fake_loss,
        }


class SegLearner(Learner):

    def __init__(self, model: ModuleFlow, meter: MetricFlow, optim, **kwargs):
        super().__init__(model, meter, optim, **kwargs)

    def match_prediction_size(self, outputs, data):
        if 'label' in data:
            shape = data['label'].shape[1:]
        else:
            shape = data['image'].shape[2:]

        # crop prediction if size mismatched
        if outputs['prediction'].shape[2:] != shape:
            outputs['prediction'] = outputs['prediction'][
                crop_range(outputs['prediction'].shape[2:], shape)
            ]

    def learn(self, data):
        outputs = self._model_run(data, training=True)
        self.match_prediction_size(outputs, data)
        data.update(outputs)
        results = self._evaluate(data, training=True)
        self._backpropagation(results['loss'])
        return results

    def _include_prediction(self, data, results):
        with torch.set_grad_enabled(False):
            probas = F.softmax(data['prediction'], dim=1)
            results.update({'prediction': probas})

    def _compute_match(self, data, results):
        with torch.set_grad_enabled(False):
            match, total = match_up(
                data['prediction'],
                data['label'],
                needs_softmax=True,
                batch_wise=True,
                threshold=-1,
            )
            results.update({'match': match, 'total': total})

    def infer(self, data, include_prediction=False, compute_match=False):
        outputs = self._model_run(data, training=False)
        self.match_prediction_size(outputs, data)
        data.update(outputs)
        results = self._evaluate(data, training=False)
        if include_prediction:
            self._include_prediction(data, results)
        if compute_match:
            self._compute_match(data, results)
        return results


class SegDisLearner:

    def __init__(
        self,
        models,
        optims,
        meters,
        grad_accumulation=1,
    ):
        # sanity check
        for key in ['seg', 'dis']:
            assert key in models, key
            assert key in optims, key

        for key in ['seg', 'dis', 'adv', 'vae']:
            assert key in meters

        # TODO: improve it
        self.training_rules = {
            'normal': ['seg', 'vae'],
            'adv': ['seg', 'vae', 'adv'],
            'ssl': ['vae', 'adv']
        }

        self.models = models
        self.meters = meters
        self.optims = optims
        self.step = 0

        for key in self.models:
            self.models[key].zero_grad()

        self.grad_accumulation = grad_accumulation
        assert isinstance(self.grad_accumulation, int)
        assert self.grad_accumulation >= 1

    def _backpropagation(self, model_name, loss):
        if self.grad_accumulation >= 1:
            loss = loss / self.grad_accumulation
        loss.backward()
        self.step += 1

        if self.step % self.grad_accumulation == 0:
            self.optims[model_name].step()
            self.models[model_name].zero_grad()

    def _dis_run(self, data, training=True):
        mask = (data['label'] >= 0).unsqueeze(1)

        if not torch.any(mask):
            return {
                'DIS_TRUTH': 0.,
                'DIS_FAKE': 0.,
            }

        else:
            # Turn on dis if needed
            if training:
                for param in self.models['dis'].parameters():
                    param.requires_grad = True
                self.models['dis'].train()
            else:
                self.models['dis'].eval()

            # on ground truth
            n_classes = data['prediction'].shape[1]
            onehot_label = F.one_hot(data['label'], n_classes)
            onehot_label = onehot_label.permute((0, 4, 1, 2, 3))
            onehot_label = onehot_label.float()

            # dis produce confidence_map
            outputs = self.models['dis']({'label': onehot_label})
            truth_loss = self.meters['dis']({
                'confidence_map': outputs['confidence_map'][mask],
                'truth': True
            })['loss']

            if training:
                self._backpropagation('dis', truth_loss)

            # on model prediction
            probas = F.softmax(data['prediction'].detach(), dim=1)
            outputs = self.models['dis']({'label': probas})
            fake_loss = self.meters['dis']({
                'confidence_map': outputs['confidence_map'][mask],
                'truth': False
            })['loss']

            if training:
                self._backpropagation('dis', fake_loss)

            return {
                'DIS_TRUTH': truth_loss,
                'DIS_FAKE': fake_loss,
            }

    def learn(self, data, mode='normal'):
        assert mode in self.training_rules

        # sanity check
        if mode in ['noraml', 'adv']:
            assert 'label' in data
        if mode == 'ssl':
            assert 'label' not in data

        with torch.set_grad_enabled(True):

            # run segmentation
            self.models['seg'].train()
            data.update(self.models['seg'](data))

            if mode in ['adv', 'ssl']:
                # post-process with softmax
                probas = torch.softmax(data['prediction'], dim=1)

                # Turn off dis
                for param in self.models['dis'].parameters():
                    param.requires_grad = False

                # dis produce confidence_map
                self.models['dis'].eval()
                data.update(self.models['dis']({'label': probas}))

            # evaluate the performance of seg
            results = {}
            loss = None
            for key in self.training_rules[mode]:
                result = self.meters[key](data)

                # sum loss
                if loss is None:
                    loss = result.pop('loss')
                else:
                    loss = loss + result.pop('loss')

                # isolate those accus except the seg accu
                if 'accu' in result and key != 'seg':
                    results.update({
                        '%s_accu' % key: result.pop('accu')
                    })

                results.update(result)
            results.update({'loss': loss})

            # finally do backpropagation on segmentor
            self._backpropagation('seg', loss)

            # run training of discriminator only on 'adv' mode
            if mode == 'adv':
                results.update(self._dis_run(data, training=True))

        return results

    def infer(self, data, include_prediction=False, compute_match=False):
        # XXX: assume mode is adv
        mode = 'adv'

        with torch.set_grad_enabled(False):
            self.models['seg'].eval()
            data.update(self.models['seg'](data))

            # post-process with softmax
            probas = torch.softmax(data['prediction'], dim=1)

            # dis produce confidence_map
            self.models['dis'].eval()
            data.update(self.models['dis']({'label': probas}))

            # evaluate the performance of seg
            results = {}
            loss = None
            for key in self.training_rules[mode]:
                result = self.meters[key](data)
                if loss is None:
                    loss = result.pop('loss')
                else:
                    loss = loss + result.pop('loss')
                results.update(result)
            results.update({'loss': loss})

            # evaluate the performance of dis
            results.update(self._dis_run(data, training=False))

            if include_prediction:
                results.update({'prediction': probas})

            if compute_match:
                match, total = match_up(
                    data['prediction'],
                    data['label'],
                    needs_softmax=True,
                    batch_wise=True,
                    threshold=-1,
                )
                results.update({'match': match, 'total': total})

        return results


class AdvSegLearner(SegLearner):

    def __init__(
        self,
        model: ModuleFlow,
        meter: MetricFlow,
        optim,
        discriminator: ModuleFlow,
        meter_unlabeled: MetricFlow = None,
        **kwargs,
    ):
        super().__init__(model, meter, optim, **kwargs)
        self.discriminator = discriminator
        self.meter_unlabeled = meter_unlabeled

    def toggle_discriminator(self, toggle=False):
        for param in self.discriminator.parameters():
            param.requires_grad = toggle

    def learn(self, data):
        self.toggle_discriminator(False)
        outputs = self._model_run(data, training=True)
        self.match_prediction_size(outputs, data)
        data.update(outputs)

        # add adversarial loss
        model_prediction = torch.softmax(outputs['prediction'], dim=1)
        outputs_dis = self.discriminator({'label': model_prediction})
        data.update(outputs_dis)
        self.toggle_discriminator(True)

        results = self._evaluate(data, training=True)
        self._backpropagation(results['loss'])
        return results

    def learn_unlabeled(self, data):
        assert 'label' not in data
        assert self.meter_unlabeled is not None

        self.toggle_discriminator(False)
        outputs = self._model_run(data, training=True)
        self.match_prediction_size(outputs, data)
        data.update(outputs)

        # preprocess with softmax before fed into discriminator
        model_prediction = torch.softmax(outputs['prediction'], dim=1)
        outputs_dis = self.discriminator({'label': model_prediction})
        data.update(outputs_dis)
        self.toggle_discriminator(True)

        with torch.set_grad_enabled(True):
            results = self.meter_unlabeled(data)
        self._backpropagation(results['loss'])
        return results

    def infer(self, data, include_prediction=False, compute_match=False):
        outputs = self._model_run(data, training=False)
        self.match_prediction_size(outputs, data)
        data.update(outputs)
        outputs_dis = self.discriminator({'label': outputs['prediction']})
        data.update(outputs_dis)
        results = self._evaluate(data, training=False)

        if include_prediction:
            self._include_prediction(data, results)
        if compute_match:
            self._compute_match(data, results)
        return results
