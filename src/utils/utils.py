import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import os
import nibabel as nib
from prettytable import from_csv
import json
import yaml
from typing import Dict
import models
import torch
import sys


class ModelHandler:

    def __init__(
        self,
        model_config,
        checkpoint=None,
    ):
        model_name = model_config.pop('name')
        self.model = getattr(models, model_name)(**model_config)

        # load checkpoint
        if checkpoint is None:
            self.checkpoint = None
        else:
            print('===== Loading checkpoint %s =====' % checkpoint)
            self.checkpoint = torch.load(
                checkpoint,
                map_location=lambda storage, location: storage
            )

            # check validity between fresh model state and pretrained one
            model_state = self.model.state_dict()
            pretrained_weights = dict()
            for key, val in self.checkpoint['model_state_dict'].items():
                if key not in model_state:
                    print('Exclude %s since not in current model state' % key)
                else:
                    if val.size() != model_state[key].size():
                        print('Exclude %s due to size mismatch' % key)
                    else:
                        pretrained_weights[key] = val

            model_state.update(pretrained_weights)
            self.model.load_state_dict(model_state)

        # swicth between multi_gpu/single gpu modes
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            self.model = self.model.cuda()

    def save(self, file_path, additional_info=dict()):
        # check if multiple gpu model
        if torch.cuda.device_count() > 1:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        content = {'model_state_dict': model_state_dict}
        if len(additional_info) >= 1:
            content.update(additional_info)
        torch.save(content, file_path)


class ROIScoreWriter:

    def __init__(self, log_file: str, ROIs: list):
        self.log_file = log_file
        self.ROIs = ROIs

        # create log file and add the header
        with open(log_file, 'w') as f:
            f.write(','.join(['Epochs'] + ROIs + ['Average']))
            f.write('\n')

    def write(self, epoch: int, roi_score: Dict[str, float]):
        # compute average score
        avg_score: float = 0.0
        for roi in self.ROIs:
            avg_score += roi_score[roi]
        avg_score /= len(self.ROIs)

        # log into the file
        info = ['%03d' % (epoch+1)]
        info += ['%.5f' % roi_score[roi] for roi in self.ROIs]
        info += ['%.5f' % avg_score]
        with open(self.log_file, 'a') as f:
            f.write(','.join(info))
            f.write('\n')


def crop_range(prediction_shape, label_shape):
    assert len(prediction_shape) == 3
    assert len(label_shape) == 3
    for p, l in zip(prediction_shape, label_shape):
        assert p >= l
    crop_range = (slice(None), slice(None))
    crop_range += tuple(
        slice((p-l) // 2, l + (p-l) // 2)
        for p, l in zip(prediction_shape, label_shape)
    )
    return crop_range


def load_config(config_file):
    file_ext = os.path.splitext(config_file)[-1].split('.')[-1]
    with open(config_file) as f:
        if file_ext == 'json':
            return json.load(f)
        elif file_ext == 'yaml':
            return yaml.safe_load(f)
        else:
            raise ValueError


def score_dict_to_markdown(score_dict):
    table = '| ID | Score | \n | - |- |'
    for key in score_dict:
        table += '\n| %s | %.5f |' % (key, score_dict[key])
    return table


def get_tty_columns():
    if sys.stdout.isatty():
        return 112
    else:
        rows, columns = os.popen('stty size', 'r').read().split()
        return int(columns)


# TODO: move epoch + 1 -> epoch, remember to modify train.py and simple_run.py
def epoch_info(epoch, total_epochs, sep='-'):
    n_cols = get_tty_columns()
    info = '%s Epoch %02d/%d %s' % (sep, epoch+1, total_epochs, sep)
    info = info + sep * ((n_cols - len(info))//2)
    info = sep * (n_cols - len(info)) + info
    print(info)


def compute_weights(npz_dir, data_list):
    def count(data_idx):
        npz_file = os.path.join(npz_dir, data_idx + '.npz')
        label = np.load(npz_file, mmap_mode='r')['label']
        label = label.reshape((-1, label.shape[-1]))

        # counts of each class
        return np.sum(label, axis=0)

    with Pool() as pool:
        counts = list(tqdm(pool.imap(count, data_list), total=len(data_list)))

    counts = np.mean(np.asarray(counts), axis=0)
    s = np.sum(counts)
    weights = [s/c for c in counts]
    return weights


def categorical_dice_score(label, pred):
    # label & prediction are both in one hot
    assert label.shape == pred.shape
    n_classes = label.shape[-1]

    # reshape for the computation
    pred = pred.reshape((-1, n_classes))
    label = label.reshape((-1, n_classes))

    # score for each class: [n_classes]
    score = 2 * np.sum(pred * label, axis=0)
    score /= np.sum(pred + label, axis=0)

    return score


def scores_to_csv(scores):
    '''
    scores: {
        data_idx: {
            ROI: score
        }
    }
    '''
    indices = list(scores.keys())
    ROIs = list(scores[indices[0]].keys())
    header = ','.join(['ID'] + ROIs + ['AVG'])

    content = ''
    for idx in indices:
        line = [idx]
        line += ['%.5f' % scores[idx][roi] for roi in ROIs]
        line += ['%.5f' % (sum(scores[idx][roi] for roi in ROIs) / len(ROIs))]
        content += '\n' + ','.join(line)
    return header + content


def csv_to_table(csv_file):
    table_file = os.path.splitext(csv_file)[0] + '.txt'
    with open(csv_file, 'r') as f1:
        with open(table_file, 'w') as f2:
            f2.write(from_csv(f1).get_string())


def save_nifti(image, nifti_file):
    assert len(image.shape) == 3, image.shape
    if np.issubdtype(image.dtype, np.floating):
        image *= 255
    image = image.astype(np.int16)
    nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
    nib.save(nifti_image, nifti_file)


def run_length_encode(label):
    # ref: www.kaggle.com/lifa08/run-length-encode-and-decode

    # make sure label containts only 0 and 1
    label = label.flatten()
    label = np.concatenate([[0], label, [0]])
    runs = np.where(label[1:] != label[:-1])[0]
    runs[1::2] -= runs[::2]

    encode = ' '.join(str(x) for x in runs)
    return encode
