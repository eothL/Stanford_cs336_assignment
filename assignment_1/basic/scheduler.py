import math

def learning_rate_schedule(t: int, lr_min: int, lr_max: int, Tw: int, Tc: int):
    assert Tc > Tw
    if t < Tw:
        return t/Tw*lr_max
    if t > Tc:

        return lr_min
    else:
        return lr_min+ (1/2) * (1 + math.cos((t-Tw) * math.pi/(Tc-Tw))) * (lr_max - lr_min)
    