import os
import random
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_all_rng(seed=None):
    """
    Random Number Generator of seed torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
                os.getpid()
                + int(datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
        )
        # print("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    set_deterministic(deterministic=True)


def set_deterministic(deterministic=True, deterministic_algorithms=False):
    cudnn.deterministic = deterministic
    cudnn.benchmark = not deterministic
    # note: line below will likely produce failures during training (some ops do not have deterministic implementation)
    # fully deterministic training launch is not possible anyway
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(deterministic)

