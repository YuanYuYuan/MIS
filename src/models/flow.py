class Flow:

    def __init__(self, config: dict, cls):
        self.nodes = {
            node_name: cls(**node_config)
            for node_name, node_config in config['nodes'].items()
        }

        for node in config['nodes']:
            assert node in config['links']['flow']
        self.links = config['links']

    def __call__(self, x, nodes=None):

        if nodes is None:
            nodes = self.nodes

        state = [None] * (self.links['outs'][-1] + 1)
        if not isinstance(x, list):
            x = [x]
        assert len(self.links['inps']) == len(x)

        for idx in self.links['inps']:
            state[idx] = x[idx]

        for node, flow in self.links['flow'].items():
            tmp = nodes[node]([state[idx] for idx in flow['inps']])
            for tmp_idx, state_idx in enumerate(flow['outs']):
                state[state_idx] = tmp[tmp_idx]

        output = [state[idx] for idx in self.links['outs']]
        # TODO: fix list type of single output
        return output[0]
