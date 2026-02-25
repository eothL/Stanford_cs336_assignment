import math


def learning_rate_schedule(t: int, lr_min: int, lr_max: int, Tw: int, Tc: int):
    assert Tc > Tw
    if t < Tw:
        return t/Tw*lr_max
    if t > Tc:

        return lr_min
    else:
        return lr_min+ (1/2) * (1 + math.cos((t-Tw) * math.pi/(Tc-Tw))) * (lr_max - lr_min)


def ramp_relu_alpha(t: int, Tw: int, Tc: int) -> float:
    """
    Cosine ramp for ReLU -> ReLU^2 interpolation.
    - warmup phase [0, Tw): alpha = 0
    - cosine phase [Tw, Tc]: alpha increases smoothly from 0 to 1
    - after cycle (t > Tc): alpha = 1
    """
    assert Tc > Tw
    if t < Tw:
        return 0.0
    if t > Tc:
        return 1.0
    phase = (t - Tw) / (Tc - Tw)
    return 0.5 * (1.0 - math.cos(math.pi * phase))
