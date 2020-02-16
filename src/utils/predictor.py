import torch
import os
import torch.nn.functional as F
from .utils import save_nifti, get_tty_columns
from tqdm import tqdm
from metrics import compute_dice, match_up
from typing import Dict


class Predictor:

    def __init__(
        self,
        model,
        load_checkpoint=None,
        save_prediction=None,
        save_data=None,
        threshold=0.2,
    ):
        self.use_cuda = torch.cuda.device_count() > 0
        self.model = model

        # load checkpoint
        if load_checkpoint is not None:
            print('===== Loading checkpoint %s =====' % load_checkpoint)
            checkpoint = torch.load(
                load_checkpoint,
                map_location=lambda storage, location: storage
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # switch between multi_gpu/single gpu/CPU only modes
            if self.use_cuda:
                if torch.cuda.device_count() > 1:
                    self.model = torch.nn.DataParallel(self.model).cuda()
                else:
                    self.model = self.model.cuda()

        # save prediction
        if save_prediction:
            os.makedirs(save_prediction, exist_ok=True)
            self.save_prediction = save_prediction
        else:
            self.save_prediction = None

        # save preprocessed data
        if save_data:
            os.makedirs(os.path.join(save_data, 'images'), exist_ok=True)
            os.makedirs(os.path.join(save_data, 'labels'), exist_ok=True)
            self.save_data = save_data
        else:
            self.save_data = None

        self.threshold = threshold

    def predict(self, batch):
        with torch.set_grad_enabled(False):
            self.model.eval()

            images = batch['image']
            labels = batch['label']
            if self.use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            logits = self.model(images)
            probas = F.softmax(logits, dim=1)

            # enhance prediction by setting a threshold
            for i, threshold in enumerate([0.9, self.threshold]):
                probas[:, i] += (probas[:, i] >= threshold).float()

            # convert probabilities into one-hot: [B, C, ...]
            max_idx = torch.argmax(probas, 1, keepdim=True)
            one_hot = torch.zeros(probas.shape)
            if self.use_cuda:
                one_hot = one_hot.cuda()
            one_hot.scatter_(1, max_idx, 1)

            match, total = match_up(
                one_hot,
                labels,
                needs_softmax=False,
                batch_wise=True
            )
            result = {'match': match, 'total': total}
            if self.save_prediction:
                result['output'] = one_hot
            return result

    def run(
        self,
        data_gen,
        verbose=False,
    ) -> Dict[str, Dict[str, float]]:

        PG = data_gen.struct['PG']
        DL = data_gen.struct['DL']
        BG = data_gen.struct['BG']

        # ensure the order
        if PG.n_workers > 1:
            assert PG.ordered
        assert BG.n_workers == 1

        # Use data list from PG since may be shuffled
        progress_bar = tqdm(
            zip(PG.data_list, PG.partition),
            total=len(PG.data_list),
            ncols=get_tty_columns(),
            desc='[Predicting] ID: %s, Accu: %.5f'
            % ('', 0.0)
        )

        # initialize necessary variables
        each_roi_score = dict()
        partition_counter = 0
        results = {
            'match': torch.Tensor(),
            'total': torch.Tensor()
        }
        if self.save_prediction:
            results['output'] = torch.Tensor()

        # use with statement to make sure data_gen is clear in case interrupted
        with data_gen as gen:
            for (data_idx, partition_per_data) in progress_bar:

                # loop until a partition have been covered
                while partition_counter < partition_per_data:
                    try:
                        batch = next(gen)
                        new_batch_result = self.predict(batch)
                    except StopIteration:
                        break

                    # append to each torch tensor in results dictionary
                    for key in new_batch_result:
                        if len(results[key]) == 0:
                            results[key] = new_batch_result[key]
                        else:
                            results[key] = torch.cat([
                                results[key],
                                new_batch_result[key]
                            ])
                    partition_counter += data_gen.struct['BG'].batch_size

                # summation over batch and combine dice score, output: [C]
                roi_score = compute_dice(
                    torch.sum(results['match'][:partition_per_data], 0),
                    torch.sum(results['total'][:partition_per_data], 0),
                )
                # exclude background
                roi_score = roi_score[1:]

                # save prediction
                if self.save_prediction:
                    one_hot_ouptuts = results['output'][:partition_per_data]
                    prediction = PG.restore(
                        data_idx,
                        torch.argmax(one_hot_ouptuts, 1)
                    )
                    file_path = os.path.join(
                        self.save_prediction,
                        data_idx + '.nii.gz'
                    )
                    save_nifti(prediction.data.cpu().numpy(), file_path)

                # save preprocessed data
                if self.save_data:
                    image_path = os.path.join(
                        self.save_data,
                        'images',
                        data_idx + '.nii.gz'
                    )
                    label_path = os.path.join(
                        self.save_data,
                        'labels',
                        data_idx + '.nii.gz'
                    )
                    save_nifti(DL.get_image(data_idx), image_path)
                    save_nifti(DL.get_label(data_idx), label_path)

                # remove processed results
                for key in results:
                    results[key] = results[key][partition_per_data:]
                partition_counter -= partition_per_data

                # record the score details
                each_roi_score[data_idx] = {
                    roi: score.item() for
                    (roi, score) in zip(DL.ROIs, roi_score)
                }

                # show progress
                progress_bar.set_description(
                    '[Predicting]  ID: %s, Accu: %.5f'
                    % (data_idx, roi_score.mean())
                )

        return each_roi_score
