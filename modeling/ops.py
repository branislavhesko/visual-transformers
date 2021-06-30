from collections import OrderedDict

import torch


def apply_ema(teacher: torch.nn.Module, student: torch.nn.Module, decay: float) -> torch.nn.Module:
    t_dict = teacher.state_dict()
    s_dict = student.state_dict()
    t_dict_new = OrderedDict()
    for name, params in s_dict.items():
        if name in t_dict:
            t_dict_new[name] = decay * t_dict[name] + (1 - decay) * params

    teacher.load_state_dict(t_dict_new)
    return teacher
