# TODO! Rename module name and create the sorresponding class (solverODE) ?
# TODO! Add scipy ivp solver method for multiple initial conditions.

def eul(f, r0, t_min, t_max, t_poi=2):
    h = (t_max - t_min) / (t_poi - 1)
    t = t_min
    r = r0.copy()

    for _ in range(1, t_poi):
        r+= h * f(r, t)
        t+= h

    return r

def rk4(f, r0, t_min, t_max, t_poi=2):
    h = (t_max - t_min) / (t_poi - 1)
    t = t_min
    r = r0.copy()

    for _ in range(1, t_poi):
        k1 = h * f(r, t)
        k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
        k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
        k4 = h * f(r + k3, t + h)
        r+= (k1 + k2 + k2 + k3 + k3 + k4) / 6.
        t+= h

    return r
