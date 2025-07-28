import torch
import torch.nn as nn
import torch.distributed as dist


def reduce_mean(tensor):
    """Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class TanhCode(nn.Module):
    def __init__(self, scale=1.0, eps=1e-5):
        super(TanhCode, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, code_, update_stats=False):
        return code_.tanh() if self.scale == 1 else code_.tanh() * self.scale

    def inverse(self, code):
        return code.clamp(min=-1 + self.eps, max=1 - self.eps).atanh() if self.scale == 1 \
            else (code / self.scale).clamp(min=-1 + self.eps, max=1 - self.eps).atanh()


class IdentityCode(nn.Module):
    @staticmethod
    def forward(code_, update_stats=False):
        return code_

    @staticmethod
    def inverse(code):
        return code


class NormalizedTanhCode(nn.Module):
    def __init__(self, mean=0.0, std=1.0, clip_range=1, eps=1e-5, momentum=0.001):
        super(NormalizedTanhCode, self).__init__()
        self.mean = mean
        self.std = std
        self.clip_range = clip_range
        self.register_buffer('running_mean', torch.tensor([0.0]))
        self.register_buffer('running_var', torch.tensor([std ** 2]))
        self.momentum = momentum
        self.eps = eps

    def forward(self, code_, update_stats=False):
        if update_stats and self.training:
            with torch.no_grad():
                var, mean = torch.var_mean(code_)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * reduce_mean(mean))
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * reduce_mean(var))
        scale = (self.std / (self.running_var.sqrt() + self.eps)).to(code_.device)
        return (code_ * scale + (self.mean - self.running_mean.to(code_.device) * scale)
                ).div(self.clip_range).tanh().mul(self.clip_range)

    def inverse(self, code):
        scale = ((self.running_var.sqrt() + self.eps) / self.std).to(code.device)
        return code.div(self.clip_range).clamp(min=-1 + self.eps, max=1 - self.eps).atanh().mul(
            self.clip_range * scale) + (self.running_mean.to(code.device) - self.mean * scale)


def build_code_activation(config):
    # TODO: make it more generic, registration
    code_act_type = config.pop('type')
    if code_act_type == "TanhCode":
        code_activation = TanhCode(**config)
    elif code_act_type == "IdentityCode":
        code_activation = IdentityCode()
    elif code_act_type == "NormalizedTanhCode":
        code_activation = NormalizedTanhCode(**config)
    else:
        raise NotImplementedError(f"Do not support code activation {code_act_type}")
    return code_activation