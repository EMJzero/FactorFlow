from factors import Coupling, Shape

coupling = Coupling(
    dims = ['X', 'Y', 'Z', 'V', 'W'],
    in_coupling = ['X', 'Y', 'Z'],
    w_coupling = ['Z', ['V', 'W']],
    out_coupling = ['X', 'Z', 'W']
    )

comp = Shape(
    X = 2048,
    Y = 4096,
    Z = 3600,
    V = 256,
    W = 1024
    )