import torch
from flows import MetricFlow, ModuleFlow
import torch.nn.functional as F
from torch.optim import Adam
import nibabel as nib
import numpy as np


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ModuleFlow('./develop/enc1.json5')
        self.enc2 = ModuleFlow('./develop/enc2.json5')
        self.seg = ModuleFlow('./develop/seg.json5')
        self.dis = ModuleFlow('./develop/dis.json5')

    def forward(self, x, switch=True):
        if switch:
            return self.seg(self.enc1(x))
        else:
            return self.seg(self.enc2(x))

class Dis(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.MaxPool3d(3),
            torch.nn.Flatten(),
            torch.nn.Linear(1000, 512),
            torch.nn.Linear(512, 128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        logits = self.model(x['latent'])
        return F.sigmoid(logits)


nets = {
    'enc1': ModuleFlow('./develop/enc1.json5'),
    'enc2': ModuleFlow('./develop/enc2.json5'),
    'seg': ModuleFlow('./develop/seg.json5'),
    # 'dis': ModuleFlow('./_develop/dis.json5'),
    'dis': Dis(),
}

optims = {
    key: Adam(nets[key].parameters(), lr=0.001)
    for key in nets
}



# meter = MetricFlow('./vae_meter.json5')

# net = Net()
# ct = nib.load('/storage/data/brain/ABCs-raw/ct/001.nii.gz').get_data()
ct = nib.load('/home/u/circle/storage/data/ABCs-raw/ct/001.nii.gz').get_data()
# t1 = nib.load('/storage/data/brain/ABCs-raw/t1/001.nii.gz').get_data()
t1 = nib.load('/home/u/circle/storage/data/ABCs-raw/t1/001.nii.gz').get_data()
# label = nib.load('/storage/data/brain/ABCs-raw/task1/001.nii.gz').get_data()
label = nib.load('/home/u/circle/storage/data/ABCs-raw/task1/001.nii.gz').get_data()


def window(image, width=100, level=50, vmin=0., vmax=1.):
    image = (image - level + width/2) * (vmax - vmin) / width + vmin
    image = np.clip(image, vmin, vmax)
    return image

def minmax_zscore(data):
    dim = 3
    axes = tuple(range(dim))
    mean = np.mean(data, axis=axes)
    std = np.std(data, axis=axes)
    data = (data - mean) / std

    # minmax
    lower_percentile = 0.2,
    upper_percentile = 99.8
    foreground = data != data[(0,) * dim]
    min_val = np.percentile(data[foreground].ravel(), lower_percentile)
    max_val = np.percentile(data[foreground].ravel(), upper_percentile)
    data[data > max_val] = max_val
    data[data < min_val] = min_val
    data = (data - min_val) / (max_val - min_val)
    data[~foreground] = 0
    return data

ct = window(ct, width=400, level=0)
t1 = minmax_zscore(t1)


crop = (slice(32, 64),) * 3
ct = ct[crop]
t1 = t1[crop]
label = label[crop]

# print(np.sum(label))

data1 = {'image': torch.Tensor(ct[np.newaxis, np.newaxis, :])}
data2 = {'image': torch.Tensor(t1[np.newaxis, np.newaxis, :])}
label = torch.Tensor(label[np.newaxis, :]).long()

# data1 = {'image': torch.rand(1, 1, 16, 16, 16)}
# data2 = {'image': torch.rand(1, 1, 16, 16, 16) * 2}
# label = torch.ones(1, 16, 16, 16).long()
# optim = Adam(net.parameters(), lr=0.001)
# # print(net(data))['prediction'].shape)

results = {}

def toggle_models(toggle):
    for key in toggle:
        if toggle[key]:
            nets[key].train()
            nets[key].training = True
        else:
            nets[key].eval()
            nets[key].training = False

for i in range(1000):

    # train enc1 + seg
    toggle_models({
        'dis': False,
        'enc1': True,
        'enc2': False,
        'seg': True
    })

    latent = nets['enc1'](data1)
    output = nets['seg'](latent)['prediction']
    loss_ce = F.cross_entropy(output, label)
    # loss_adv = 1 - nets['dis'](latent)
    # loss = loss_ce + loss_adv * 0.3
    loss = loss_ce

    for key in ['enc1', 'seg']:
        optims[key].zero_grad()
    loss.backward()
    for key in ['enc1', 'seg']:
        optims[key].step()

    results.update({
        'CE': loss_ce.detach(),
        # 'ADV': loss_adv.detach()
    })

    # align encoders
    toggle_models({
        'dis': False,
        'enc1': True,
        'enc2': True,
        'seg': False
    })
    # loss = 1 + nets['dis'](nets['enc1'](data1)) - nets['dis'](nets['enc2'](data2))
    loss =  nets['dis'](nets['enc1'](data1)) - nets['dis'](nets['enc2'](data2))
    loss = loss * loss
    for key in ['enc1', 'enc2']:
        optims[key].zero_grad()
    loss.backward()
    for key in ['enc1', 'enc2']:
        optims[key].step()

    results.update({'ENC': loss.detach()})

    # train dis
    toggle_models({
        'dis': True,
        'enc1': False,
        'enc2': False,
        'seg': False
    })
    loss = nets['dis'](nets['enc2'](data2)) - nets['dis'](nets['enc1'](data1))
    loss = -1 * loss ** 2

    for key in ['dis']:
        optims[key].zero_grad()
    loss.backward()
    for key in ['dis']:
        optims[key].step()

    results.update({'DIS': loss.detach()})

    # # da
    # toggle_models({
    #     'dis': True,
    #     'enc1': True,
    #     'enc2': True,
    #     'seg': False
    # })

    # loss = 1 + nets['dis'](nets['enc1'](data1)) - nets['dis'](nets['enc2'](data2))

    # for key in ['enc1', 'enc2', 'dis']:
    #     optims[key].zero_grad()
    # loss.backward()
    # for key in ['enc1', 'enc2', 'dis']:
    #     optims[key].step()

    # results.update({'DA': loss.detach()})

    # # verify
    # toggle_models({
    #     'dis': False,
    #     'enc1': False,
    #     'enc2': False,
    #     'seg': False
    # })

    loss1 = F.cross_entropy(nets['seg'](nets['enc1'](data1))['prediction'], label).detach()
    loss2 = F.cross_entropy(nets['seg'](nets['enc2'](data2))['prediction'], label).detach()

    results.update({'L1': loss1.detach(), 'L2': loss2.detach()})

    info = ', '.join(['%s: %.3f' % (key, results[key]) for key in results])
    print(info)
