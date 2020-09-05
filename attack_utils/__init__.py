import foolbox as fb


attack_dict = {"FGSM_inf": fb.attacks.LinfFastGradientAttack,
               "BrendelBethge_inf": fb.attacks.LinfinityBrendelBethgeAttack,
               "DeepFool_inf": fb.attacks.LinfDeepFoolAttack,
               "CW_2": fb.attacks.L2CarliniWagnerAttack,
               "FGSM_2": fb.attacks.L2FastGradientAttack,
               "FGSM_1": fb.attacks.L1FastGradientAttack,
               "DeepFool_2": fb.attacks.L2DeepFoolAttack,
               "PGD_2": fb.attacks.L2ProjectedGradientDescentAttack,
               "PGD_inf": fb.attacks.LinfProjectedGradientDescentAttack,
               "BrendelBethge_0": fb.attacks.L0BrendelBethgeAttack,
               "MIM_inf": fb.attacks.LinfBasicIterativeAttack}


class BetterDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_attack_fn(attack_config):
    config = BetterDict(attack_config)
    attack_cls = attack_dict["_".join([config.attack_name, config.constraint])]
    return attack_cls(**config.attack_args), config

