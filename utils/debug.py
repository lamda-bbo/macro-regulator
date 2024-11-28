import torch as th

def assert0(*args):
    print('')
    print('****************************Assert0****************************')
    for arg in args:
        print(arg)
    print('******************************End******************************')
    print('')
    assert(0)

def highlight(*args):
    print('')
    print('****************************HighLight****************************')
    for arg in args:
        print(arg)
    print('*******************************End*******************************')
    print('')

def hasnan(tensor:th.tensor)->bool:
    if th.isnan(tensor).any().item():
        print('Nan in tensor')
        return True
    return False

def hasinf(tensor:th.tensor)->bool:
    if th.isinf(tensor).any().item():
        print('Inf in tensor')
        return True
    return False
