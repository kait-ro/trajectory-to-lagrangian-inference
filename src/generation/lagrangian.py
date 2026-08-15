import sympy as sp
# Some lagrangian is of the form 
# L = c_1*q1 + c_2*q2+c_3*q3 and so on

# I want it so that whatever lagrangian I put in i.e
# The Lagrangian
# It goes through allat, takes all the symbolic meaning from it
# Derives the Euler lagrange equations for it
# For example if an L is of the form
# L(q, q_dot)
# So it does EL by both q and q_dot
# Then if there's a second q2, then it does q2 and q2_dot

# Yo I wanna grab some coordinates from the user

def coordinates(no_coords: int) -> list:
    t = sp.symbols('t') # creating a symbol t that python
    # as just a letter not a variable, :)
    coords = []
    for i in range(no_coords):
        