import torch.nn as nn
from . import modules
import torch


class Map(nn.Module):

    def __init__(self, name, **kwargs):
        super().__init__()
        if hasattr(modules, name):
            self.op = getattr(modules, name)(**kwargs)
        elif hasattr(nn, name):
            self.op = getattr(nn, name)(**kwargs)
        else:
            raise NotImplementedError(name)

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

        # print('=======================')

        # construct maps
        for state_out_idx, maps in self.topo.items():
            state_out_idx = int(state_out_idx)
            if isinstance(maps, str):
                state[state_out_idx] = self.ops[maps](state[state_out_idx-1])
            else:
                assert isinstance(maps, dict)
                if 'mode' in maps:
                    mode = maps['mode']
                    assert mode in ['sum', 'cat']
                else:
                    mode = 'sum'

                tmp = None
                for map_idx, state_in_idx in maps.items():
                    if map_idx == 'mode':
                        continue
                    if tmp is None:
                        tmp = self.ops[map_idx](state[state_in_idx])
                    else:
                        new = self.ops[map_idx](state[state_in_idx])
                        if mode == 'sum':
                            assert tmp.shape == new.shape, (tmp.shape, new.shape)
                            tmp = tmp + new
                        elif mode == 'cat':
                            assert tmp.shape[2:] == new.shape[2:]
                            tmp = torch.cat((tmp, new), dim=1)
                        else:
                            raise ValueError(mode)

                state[state_out_idx] = tmp

            # shape = tuple(state[state_out_idx].shape) \
            #     if isinstance(state[state_out_idx], torch.Tensor) else None
            # print('Map: {}, Idx: {}, Shape: {}'.format(
            #     maps,
            #     state_out_idx,
            #     shape
            # ))

        # print('\n\n')

        # if len(self.outs) > 1:
        #     return [state[idx] for idx in self.outs]
        # else:
        #     return state[self.outs[0]]
        return [state[idx] for idx in self.outs]
