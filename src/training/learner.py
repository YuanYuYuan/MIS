import torch
import torch.nn.functional as F
from utils import crop_range
from metrics import match_up


class Learner:

    def __init__(self, model, meter, optim):
        self.model = model
        self.meter = meter
        self.optim = optim

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
        with torch.set_grad_enabled(True):
            self.model.train()
            self.optim.zero_grad()

            outputs = self.model(data)
            self.match_prediction_size(outputs, data)
            data.update(outputs)
            results = self.meter(data)

            # back propagation
            results['loss'].backward()
            self.optim.step()

        return results

    def infer(self, data, include_prediction=False, compute_match=False):

        with torch.set_grad_enabled(False):
            self.model.eval()

            outputs = self.model(data)
            self.match_prediction_size(outputs, data)
            data.update(outputs)
            results = self.meter(data)

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
