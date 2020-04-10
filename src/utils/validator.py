import torch
import os
import torch.nn.functional as F
from .utils import save_nifti, get_tty_columns
from tqdm import tqdm
from metrics import compute_dice, match_up


class Validator:

    def __init__(
        self,
        model,
        load_checkpoint=None,
        save_output=None,
        save_input=None,
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

        # save output
        if save_output:
            os.makedirs(save_output, exist_ok=True)
            self.save_output = save_output
        else:
            self.save_output = None

        # save preprocessed data
        if save_input:
            os.makedirs(os.path.join(save_input, 'images'), exist_ok=True)
            os.makedirs(os.path.join(save_input, 'labels'), exist_ok=True)
            self.save_input = save_input
        else:
            self.save_input = None

        self.threshold = threshold

    def validate(self, batch):
        with torch.set_grad_enabled(False):
            self.model.eval()

            images = batch['image']
            labels = batch['label']
            if self.use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            logits = self.model(images)
            probas = F.softmax(logits, dim=1)

            # enhance output by setting a threshold
            for i in range(1, probas.shape[1]):
                probas[:, i, ...] += (probas[:, i, ...] >= self.threshold).float()

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
            if self.save_output:
                result['output'] = one_hot
            return result

    def run(
        self,
        data_gen,
        verbose=False,
    ):

        PG = data_gen.struct['PG']
        DL = data_gen.struct['DL']
        BG = data_gen.struct['BG']

        # Use data list from PG since may be shuffled
        data_list = PG.data_list
        ROIs = DL.ROIs

        # ensure the order
        if PG.n_workers > 1:
            assert PG.ordered
        assert BG.n_workers == 1

        progress_bar = tqdm(
            zip(data_list, PG.partition),
            total=len(data_list),
            ncols=get_tty_columns(),
            dynamic_ncols=True,
            desc='[Validating] ID: %s, Accu: %.5f'
            % ('', 0.0)
        )

        # initialize necessary variables
        each_roi_score = dict()
        partition_counter = 0
        results = {
            'match': torch.Tensor(),
            'total': torch.Tensor()
        }
        if self.save_output:
            results['output'] = torch.Tensor()

        # use with statement to make sure data_gen is clear in case interrupted
        with data_gen as gen:
            for (data_idx, partition_per_data) in progress_bar:

                # loop until a partition have been covered
                while partition_counter < partition_per_data:
                    try:
                        batch = next(gen)
                        new_batch_result = self.validate(batch)
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
                    smooth=1e-5,
                )
                # exclude background
                roi_score = roi_score[1:]

                # save output
                if self.save_output:
                    save_nifti(

                        # restore data from output
                        PG.restore(
                            data_idx,
                            torch.argmax(results['output'][:partition_per_data], 1)
                        ).data.cpu().numpy(),

                        # nifti file path
                        os.path.join(
                            self.save_output,
                            data_idx + '.nii.gz'
                        )
                    )

                # save preprocessed data
                if self.save_input:
                    save_nifti(
                        DL.get_image(data_idx),
                        os.path.join(
                            self.save_input,
                            'images',
                            data_idx + '.nii.gz'
                        )
                    )
                    save_nifti(
                        DL.get_label(data_idx),
                        os.path.join(
                            self.save_input,
                            'labels',
                            data_idx + '.nii.gz'
                        )
                    )

                # remove processed results
                for key in results:
                    results[key] = results[key][partition_per_data:]
                partition_counter -= partition_per_data

                # record the score details
                each_roi_score[data_idx] = {
                    roi: score.item() for
                    (roi, score) in zip(ROIs, roi_score)
                }

                # show progress
                progress_bar.set_description(
                    '[Validating]  ID: %s, Accu: %.5f'
                    % (data_idx, roi_score.mean())
                )

        # compute total average score
        avg_score = sum(
            sum(each_roi_score[data_idx].values()) / len(ROIs)
            for data_idx in data_list
        ) / len(data_list)

        # compute score for each roi: {roi: average score}
        roi_score = {roi: 0.0 for roi in ROIs}
        for data_idx in data_list:
            for roi in ROIs:
                roi_score[roi] += each_roi_score[data_idx][roi]
        for roi in ROIs:
            roi_score[roi] /= len(data_list)

        info = ['Avg Accu: %.5f' % avg_score]
        info += [
            '%s: %.5f' % (roi, score)
            for (roi, score) in roi_score.items()
        ]
        print(', '.join(info))

        return {
            # {data_idx: {roi: score}}
            'raw': each_roi_score,

            # {roi: avg score over data_list}
            'roi': roi_score,

            # avg over all roi and data_list
            'avg': avg_score,
        }
