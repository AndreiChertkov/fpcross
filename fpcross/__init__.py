from .config import config
from .utils import ij
from .mtr import difscheb, dif1cheb, dif2cheb
from .tns import polycheb
from .grid import Grid
from .func import Func
from .ord_solver import OrdSolver
from .solver import Solver
from .model import Model
from .check import Check

# TODO Perhaps where should be convention everywhere that the list and
#      np.ndarray arguments are always interchangeable (it will lead to the
#      more compact doc strings). Or add more compact doc string (for example,
#      type: list/arr [5] of int).

# TODO Add convention about init/prep/calc to the README file.
