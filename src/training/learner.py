import torch
from utils import crop_range
from metrics import match_up


class Learner:

    def __init__(self, model, meter, optim):
        self.model = model
        self.meter = meter
        self.optim = optim

    def learn(self, data):
        with torch.set_grad_enabled(True):
            self.model.train()
            self.optim.zero_grad()
            outputs = self.model(data)

            # crop if size mismatched
            if outputs['prediction'].shape[2:] != data['label'].shape[1:]:
                outputs['prediction'] = outputs['prediction'][
                    crop_range(
                        outputs['prediction'].shape[2:],
                        data['label'].shape[1:]
                    )
                ]

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

            # crop if size mismatched
            if outputs['prediction'].shape[2:] != data['label'].shape[1:]:
                outputs['prediction'] = outputs['prediction'][
                    crop_range(
                        outputs['prediction'].shape[2:],
                        data['label'].shape[1:]
                    )
                ]

            data.update(outputs)
            results = self.meter(data)

        if include_prediction:
            probas = torch.nn.functional.softmax(outputs['prediction'], dim=1)
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
