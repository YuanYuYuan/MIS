import torch
from flows import ModuleFlow


class ModelHandler:

    def __init__(
        self,
        model_config,
        checkpoint=None,
    ):
        self.model = ModuleFlow(model_config)

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
