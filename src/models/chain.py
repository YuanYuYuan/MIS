import torch.nn as nn


class Map(nn.Module):

    def __init__(
        self,
        name,
        ch_in=16,
        ch_out: int = None,
        postprocess=True,
    ):
        super().__init__()

        self.op = nn.Sequential()
        self.ch_in = ch_in
        self.ch_out = ch_in if ch_out is None else ch_out
        self.postprocess = postprocess

        if name == '2D':
            self.op.add_module(
                '2D',
                nn.Conv3d(
                    self.ch_in,
                    self.ch_out,
                    kernel_size=(3, 3, 1),
                    padding=(1, 1, 0),
                    bias=False
                )
            )

        elif name == '3D':
            self.op.add_module(
                '3D',
                nn.Conv3d(
                    self.ch_in,
                    self.ch_out,
                    kernel_size=3,
                    padding=1,
                    bias=False
                )
            )

        elif name == 'P3D':
            self.op.add_module(
                'P3D_1',
                nn.Conv3d(
                    self.ch_in,
                    self.ch_out,
                    kernel_size=(3, 3, 1),
                    padding=(1, 1, 0),
                    bias=False
                )
            )
            self.op.add_module(
                'P3D_2',
                nn.Conv3d(
                    self.ch_in,
                    self.ch_out,
                    kernel_size=(1, 1, 3),
                    padding=(0, 0, 1),
                    bias=False
                )
            )

        elif name == 'Downsample':
            self.ch_out = self.ch_in * 2
            self.op.add_module(
                'Downsample',
                nn.Conv3d(
                    self.ch_in,
                    self.ch_out,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False
                ),
            )

        elif name == 'Upsample':
            self.ch_out = self.ch_in // 2
            self.op.add_module(
                'Upsample_1',
                nn.Upsample(scale_factor=2, mode="trilinear"),
            )
            self.op.add_module(
                'Upsample_2',
                nn.Conv3d(
                    self.ch_in,
                    self.ch_out,
                    kernel_size=1,
                    bias=False
                )
            )

        elif name == 'Identity':
            self.op.add_module('Identity', nn.Identity())

        else:
            raise NotImplementedError
        if postprocess and name != 'Identity':
            self.op.add_module(
                'norm',
                nn.InstanceNorm3d(self.ch_out, affine=True)
            )
            self.op.add_module('acti', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.op(x)


class Chain(nn.Module):

    def __init__(
        self,
        inps: list,
        outs: list,
        topo: dict,
        maps: dict,
    ):
        super().__init__()

        self.ops = nn.ModuleDict({
            op_name: Map(**map_config)
            for(op_name, map_config) in maps.items()
        })
        self.topo = topo
        self.outs = outs
        self.inps = inps
        self.n_states = max(self.outs) + 1

    def forward(self, input_state):

        # initialize state
        state = [None] * self.n_states
        if not isinstance(input_state, list):
            input_state = [input_state]
        for idx, s in zip(self.inps, input_state):
            state[idx] = s

        # construct maps
        for state_out_idx, maps in self.topo.items():
            state_out_idx = int(state_out_idx)
            if isinstance(maps, str):
                state[state_out_idx] = self.ops[maps](state[state_out_idx-1])
            else:
                tmp = None
                for map_idx, state_in_idx in maps.items():
                    if tmp is None:
                        tmp = self.ops[map_idx](state[state_in_idx])
                    else:
                        tmp = tmp + self.ops[map_idx](state[state_in_idx])
                state[state_out_idx] = tmp

        if len(self.outs) > 1:
            return [state[idx] for idx in self.outs]
        else:
            return state[self.outs[0]]
