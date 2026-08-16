import sympy as sp


def define_coordinates(no_coords: int) -> list:
    t = sp.symbols("t")  # creating a symbol t that python
    # as just a letter not a variable, :)
    coords = []
    vels = []
    for i in range(no_coords):
        coords.append(sp.Function(f"q{i}")(t))
    for j in coords:
        vels.append(sp.diff(j, t))
    return t, coords, vels


def define_lagrangian(coords: list, vels: list):
    (q_dot,) = vels
    (q,) = coords
    m, g, l = sp.symbols("m g l")
    L = sp.Rational(1, 2) * m * q_dot**2 + m * g * l * sp.cos(q)
    return L
