"""Demo script that shows how to use the Fokker-Planck solver.

Run it from the root of the project as "clear && python demo/demo.py". The
output should look like this (see also the plot "demo/demo_result.png"):
"
Solve: 100step [00:06, 16.33step/s,  | t=2.50 | r=1.0 | e_t=1.50e-02]

The value of PDF at poi=[0, 0, 0] for different time moments:
Time   0.0250 | Value   0.0433
Time   0.0500 | Value   0.0423
Time   0.0750 | Value   0.0413
...
Time   2.4750 | Value   0.0095
Time   2.5000 | Value   0.0094
".

Note:
    To create the new equation, the base (abstract) class "Equation" should be
    inherited (see the class "EquationOwn" below). The detailed description of
    the base class methods and the format of input and output arguments is
    presented in the module "fpcross/equation.py" (online documentation will be
    available soon).

    Numerous parameters of the TT-CROSS method can be refined when calling the
    "Equation" class method "set_cross_opts". See the documentation for this
    function for more details. At the same time, for all prepared model
    examples (multidimensional simple diffusion problem, multidimensional
    Ornstein-Uhlenbeck process and 3-dimensional dumbbell model), a high quality
    solution is obtained with the default values of these parameters.

    For the methods "f", "f1", "r0", "rs", "rt" of the "Equation" class, input
    variable "X" is np.ndarray of the shape [samples, dimensions]. The output
    should be np.ndarray of the shape [samples, dimensions] for "f" and "f1",
    and np.ndarray of the shape [samples] for "r0", "rs" and "rt".

"""
import numpy as np


from fpcross import Equation
from fpcross import FPCross


class EquationOwn(Equation):
    """Fokker-Planck equation, which relates to the simple diffusion problem."""

    def callback(self, solver):
        """Function is called by solver after each time step."""

        # We calculate the value of the solution at some point at each
        # time step (just for demonstration purposes):
        self.t_list.append(solver.t)
        self.y_list.append(solver.get(self.poi))

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
        self.t_list = []
        self.y_list = []

    def r0(self, X):
        """Initial PDF r(x, 0). The Gaussian distribution is used."""
        a = 2. * self.coef_pdf
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

    def set_coef_pdf(self, coef_pdf=1.):
        """Set coefficient for initial PDF Gaussian distribution.

        This function is presented in the base class "Equation" and it is called
        automatically by class constructor. We present it here for
        demonstration purposes only. Note that the value of this parameter is
        used in both "r0" and "rt" methods.

        """
        self.coef_pdf = coef_pdf

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
fpc = FPCross(
    eq,
    with_hist=True) # Accuracy will be checked after each step with this flag
fpc.solve()
fpc.plot('./demo/demo_result.png')

# We may print the computed values in selected point from each time step:
print('\nThe value of PDF at poi=[0, 0, 0] for different time moments:')
for t, y in zip(eq.t_list, eq.y_list):
    print(f'Time {t:-8.4f} | Value {y:-8.4f}')
