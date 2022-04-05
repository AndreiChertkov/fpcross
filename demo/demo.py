"""Demo script that shows how to use the Fokker-Planck solver.

To create the new equation, the base (abstract) class "Equation" should be
inherited. See the detailed description of the base class methods and the
format of input and output arguments in the module "fpcross/equation.py".

"""
import numpy as np


from fpcross import Equation
from fpcross import FPCross


class EquationOwn(Equation):
    def callback(self, solver):
        """Function is called by solver after each time step."""

        # We calculate the value of the solution at some point at each
        # time step (just for demonstration purposes):
        v = solver.get(self.poi)
        self.val.append(v)

    def f(self, X, t):
        """The rhs."""
        return np.zeros(X.shape)

    def f1(self, X, t):
        """The diagonal derivatives for the rhs."""
        return np.zeros(X.shape)

    def init(self):
        """Initialize the equation parameters."""
        self.with_rs = False # Should be True, if "rs" function is set
        self.with_rt = True  # Should be True, if "rt" function is set

        # We will calculate the value of the solution at this point at each
        # time step (just for demonstration purposes):
        self.poi = np.zeros(self.d)
        self.val = []

    def r0(self, X, coef_pdf=1.):
        """Initial PDF r(x, 0)."""
        a = 2. * coef_pdf
        r = np.exp(-np.sum(X*X, axis=1) / a)
        r /= (np.pi * a)**(self.d/2)
        return r

    def rs(self, X):
        """Stationary PDF r0(x) for accuracy check."""
        return

    def rt(self, X, t):
        """Time-dependent real PDF r(x, t) for accuracy check."""
        a = 2. * self.coef_pdf + 4. * self.coef * t
        r = np.exp(-np.sum(X*X, axis=1) / a)
        r /= (np.pi * a)**(self.d/2)
        return r

# First we have to create a description of the equation to be solved:
eq = EquationOwn(
    d=3,                          # Number of dimensions.
    e=1.E-6,                      # Desired accuracy of approximation (for TT)
    is_full=False,                # Use full (numpy) format or TT-format
    name='My equation')           # Display name
eq.set_grid(n=20, a=-10., b=+10.) # Number of spatial points, lower/upper bounds
eq.set_grid_time(m=100, t=2.5)    # Number of time points, upper bound for time
eq.init()                         # Call it after setting all equation params

# Now we can solve the eqation:
fpc = FPCross(eq, with_hist=True)
fpc.solve()
fpc.plot('./demo/demo_result.png')

# We may print the computed values in selected point at each time step:
print(eq.val)
