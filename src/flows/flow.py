class Flow:

    def __init__(self, config: dict, cls: type):
        self.nodes = {
            node_name: cls(**node_config)
            for node_name, node_config in config['nodes'].items()
        }

        for node in config['nodes']:
            assert node in config['links']['flow']
        self.links = config['links']

        if isinstance(self.links['outs'], dict):
            self.n_states = max(self.links['outs'].values()) + 1
        else:
            self.n_states = max(self.links['outs']) + 1

    def __call__(self, x, nodes=None):

        if nodes is None:
            nodes = self.nodes

        state = [None] * self.n_states
        for key, idx in self.links['inps'].items():
            assert key in x, (key, list(x.keys()))
            state[idx] = x[key]

        # if not isinstance(x, list):
        #     x = [x]
        # assert len(self.links['inps']) == len(x)

        # for idx in self.links['inps']:
        #     state[idx] = x[idx]

        for node, flow in self.links['flow'].items():
            tmp = nodes[node]([state[idx] for idx in flow['inps']])
            for tmp_idx, state_idx in enumerate(flow['outs']):
                state[state_idx] = tmp[tmp_idx]

        # XXX
        # return {key: state[idx] for key, idx in self.links['outs'].items()}

        # XXX
        if 'unwrap' in self.links['outs']:
            return state[self.links['outs']['unwrap']]
        else:
            return {key: state[idx] for key, idx in self.links['outs'].items()}
