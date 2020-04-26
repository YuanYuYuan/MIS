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

        label_mask = (data['label'] >= 0).unsequeeze(1)
        if not any(label_mask):
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

        # preprocess with softmax before feeding into discriminator
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
