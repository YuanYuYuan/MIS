import torch
import torch.nn.functional as F
from utils import crop_range
from metrics import match_up
from flows import MetricFlow, ModuleFlow


class Learner:

    def __init__(self, model: ModuleFlow, meter: MetricFlow, optim):
        self.model = model
        self.meter = meter
        self.optim = optim

    def _model_run(self, data, training=True):

        if training:
            with torch.set_grad_enabled(True):
                self.model.train()
                self.optim.zero_grad()
                return self.model(data)
        else:
            with torch.set_grad_enabled(False):
                self.model.eval()
                return self.model(data)

    def _evaluate(self, data, training=True):

        if training:
            with torch.set_grad_enabled(True):
                results = self.meter(data)

                # back propagation
                results['loss'].backward()
                self.optim.step()
                return results
        else:
            with torch.set_grad_enabled(False):
                return self.meter(data)

    def learn(self, data):
        outputs = self._model_run(data, training=True)
        data.update(outputs)
        return self._evaluate(data, training=True)

    def infer(self, data):
        outputs = self._model_run(data, training=False)
        data.update(outputs)
        return self._evaluate(data, training=False)


class SegLearner(Learner):

    def __init__(self, model: ModuleFlow, meter: MetricFlow, optim):
        super().__init__(model, meter, optim)

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
        return self._evaluate(data, training=True)

    def infer(self, data, include_prediction=False, compute_match=False):
        outputs = self._model_run(data, training=False)
        self.match_prediction_size(outputs, data)
        data.update(outputs)
        results = self._evaluate(data, training=False)
        with torch.set_grad_enabled(False):
            if include_prediction:
                probas = F.softmax(outputs['prediction'], dim=1)
                results.update({'prediction': probas})

            if compute_match:
                match, total = match_up(
                    outputs['prediction'],
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
    ):
        super().__init__(model, meter, optim)
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
        outputs_dis = self.discriminator({'label': outputs['prediction']})
        data.update(outputs_dis)
        self.toggle_discriminator(True)
        return self._evaluate(data, training=True)

    def learn_unlabeled(self, data):
        assert 'label' not in data
        assert self.meter_unlabeled is not None

        self.toggle_discriminator(False)
        outputs = self._model_run(data, training=True)
        self.match_prediction_size(outputs, data)
        data.update(outputs)
        outputs_dis = self.discriminator({'label': outputs['prediction']})
        data.update(outputs_dis)
        with torch.set_grad_enabled(True):
            results = self.meter_unlabeled(data)

            # back propagation
            results['loss'].backward()
            self.optim.step()
        self.toggle_discriminator(True)
        return results
