"""
Skip weight decay for bias and normalization parameters. Modified from

https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136

"""
import torch.nn as nn

model_dict = {
    'linear': nn.Linear,
    'conv1d': nn.Conv1d,
    'conv2d': nn.Conv2d,
    'conv3d': nn.Conv3d,
    'layer_norm': nn.LayerNorm,
    'batch_norm_1d': nn.BatchNorm1d,
    'batch_norm_2d': nn.BatchNorm2d,
    'batch_norm_3d': nn.BatchNorm3d,
    'embedding': nn.Embedding,
}


def configure_optimizers(model, blacklist_module_names=None, only_trainable: bool = False, verbose=False):
    if blacklist_module_names is None:
        blacklist_module_names = ['layer_norm', 'batch_norm_1d', 'batch_norm_2d', 'batch_norm_3d', 'embedding']

    # separate out all parameters to those that will and won't experience regularizing weight decay
    no_decay = set()

    blacklist_weight_modules = tuple([model_dict[x] for x in blacklist_module_names])
    #  recursively walks into all modules
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            if pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    decay = {pn for pn, p in model.named_parameters() if pn not in no_decay}
    if verbose:
        print(f"no_decay params: ")
        for x in sorted(list(no_decay)):
            print(f"\t {x}")
        print(f"decay params: ")
        for x in sorted(list(decay)):
            print(f"\t {x}")

        message = "No decay params excluding bias term: \n"
        for x in no_decay:
            if not x.endswith('bias'):
                message += f"\t {x}\n"
        print(message)

        print(f"Decay params excluding weight term: ")
        for x in decay:
            if not x.endswith('weight'):
                print(f"\t {x}")

    print(f"{model.__class__.__name__} summary: {len(decay)} decay params and {len(no_decay)} params")

    # validate considered all parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, \
        f"parameters {str((param_dict.keys() - union_params))} were not separated into either decay/no_decay set!"

    if only_trainable:
        trainable_param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        if verbose and len(trainable_param_dict) != len(param_dict):
            non_trainable_params = param_dict.keys() - trainable_param_dict.keys()
            print(f"Non-trainable params:")
            for x in sorted(list(non_trainable_params)):
                print(f"\t {x}")
        optim_groups = [
            {"params": [trainable_param_dict[pn] for pn in sorted(list(decay)) if pn in trainable_param_dict]},
            {"params": [trainable_param_dict[pn] for pn in sorted(list(no_decay)) if pn in trainable_param_dict],
             "weight_decay": 0.0},
        ]
    else:
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
    return optim_groups
