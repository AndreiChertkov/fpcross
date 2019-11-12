import numpy as np
import matplotlib.pyplot as plt

class Grid(object):
    '''
    Class for representation and construction of the
    uniform or Chebyshev multidimensional grid.

    Basic usage:
    1 Initialize class instance with grid parameters.
    2 Call "comp" for computation in given indices (or the full grid).
    3 Call "info" for demonstration of grid parameters.
    4 Call "copy" to obtain new instance with the same parameters.

    Advanced usage:
    - Call "find" to find the nearest flatten grid index for the given point.
    - Call "rand" to obtain random points inside the grid limits.
    - Call "plot" for plot of some or all grid points (or random points).
    - Call "is_in" to check if given point is inside the grid.
    - Call "is_out" to check if given point is outside the grid.
    - Call "is_sym" to check if grid is symmetric (l_max = - l_min).
    - Call "is_square" to check if grid is square.

    PROPS:

    d - number of the grid dimensions
    type: int >= 1

    n - total number of points for each dimension
    type: ndarray [dimensions] of int >= 2

    l - min ([:, 0]) and max ([:, 1]) values of variable for each dimension
    type: ndarray [dimensions, 2] of float,
          [i, 0] < [i, 1] for each i

    k - kind of the grid
    type: str
    enum:
        - 'u' - uniform grid
        - 'c' - Chebyshev grid

    h - grid steps assuming uniformity for each dimension
    type: ndarray [dimensions] of float > 0,
          h[i] <= l[i, 1] - l[i, 0] for each i

    n0 - average number of grid points (mean for n)
    type: int >= 2
    * For the 1D grid it is scalar number of grid points.

    l1 - average min grid limit (mean for l[:, 0])
    type: float
    * For the 1D grid it is scalar min limit.

    l2 - average max grid limit (mean for l[:, 1])
    type: float
    * For the 1D grid it is scalar max limit.

    h0 - average grid step assuming uniformity (mean for h)
    type: float > 0 and <= l2 - l1
    * For the 1D grid it is scalar grid step (assuming uniformity).

    TODO Add function that scale points to/from limits l and [0, 1].
    '''

    def __init__(self, d=None, n=2, l=[-1., 1.], k='c'):
        '''
        INPUT:

        d - number of the grid dimensions
        type1: None
        type2: int >= 1
        * If is None (type1), then it will be recovered from n or l shape.
        * If is set (type2), then n and l will be extended (if required)
        * according to the number of dimensions.

        n - total number of points for each dimension
        type1: int >= 2
        type2: list [dimensions] of int >= 2
        type3: ndarray [dimensions] of int >= 2
        * If is int (type1), then it will be used for each dimension.

        l - min and max values of variable for each dimension
        type1: list [2] of float,
               [0] < [1]
        type2: ndarray [2] of float,
               [0] < [1]
        type3: list [dimensions, 2] of float,
               [i, 0] < [i, 1] for each i
        type4: ndarray [dimensions, 2] of float,
               [i, 0] < [i, 1] for each i
        * Note that [:, 0] (or [0]) are min and [:, 1] (or [1]) are max
        * values for each dimension. If it is 1D list or array (type1 or type2),
        * then the same values will be used for each dimension.

        k - kind of the grid
        type: str
        enum:
            - 'u' - uniform grid
            - 'c' - Chebyshev grid
        '''

        # Check / prepare d

        if d is None:
            d = 1
            if isinstance(n, (list, np.ndarray)):
                if len(n) > 0:
                    d = len(n)
            elif isinstance(l, (list, np.ndarray)):
                if len(l) > 0 and isinstance(l[0], (list, np.ndarray)):
                    d = len(l)
        else:
            if not isinstance(d, (int, float)) or d < 1:
                raise ValueError('Invalid number of dimensions (d).')
            d = int(d)

        # Check / prepare n

        if isinstance(n, (int, float)):
            n = [int(n)] * d
        if isinstance(n, list):
            n = np.array(n, dtype='int')
        if not isinstance(n, np.ndarray) or len(n.shape) != 1:
            raise ValueError('Invalid type or shape for number of points (n).')
        if n.shape[0] != d:
            raise IndexError('Invalid dimension for number of points (n).')
        for i in range(d):
            if n[i] < 2:
                raise ValueError('Ivalid number of points (should be >= 2).')
            if int(n[i]) != n[i]:
                raise ValueError('Ivalid type for number of points (n).')
        n = n.astype(int)

        # Check / prepare l

        if isinstance(l, list):
            l = np.array(l)
        if not isinstance(l, np.ndarray) or not len(l.shape) in [1, 2]:
            raise ValueError('Invalid type or shape for limits (l).')
        if len(l.shape) == 1:
            l = np.repeat(l.reshape(1, -1), d, axis=0)
        if l.shape[0] != d:
            raise IndexError('Invalid dimension for limits (l).')
        if l.shape[1] != 2:
            raise IndexError('Invalid shape for limits (l).')
        for i in range(d):
            if l[i, 0] >= l[i, 1]:
                raise ValueError('Ivalid limits (min should be less of max).')

        # Check / prepare k

        if k != 'u' and k != 'c':
            raise ValueError('Invalid grid kind.')

        # Set parameters

        self.d = d
        self.n = n
        self.l = l
        self.k = k
        self.h = (self.l[:, 1] - self.l[:, 0]) / (self.n - 1)

        # Set mean values for parameters

        self.n0 = int(np.mean(self.n))
        self.l1 = float(np.mean(self.l[:, 0]))
        self.l2 = float(np.mean(self.l[:, 1]))
        self.h0 = float(np.mean(self.h))

    def copy(self, **kwargs):
        '''
        Create a copy of the grid.

        INPUT:

        **kwargs - some arguments from Grid.__init__
        type: dict
        * These values will replace the corresponding params in the new grid.

        OUTPUT:

        GR - new class instance with the same parameters
        type: Grid
        '''

        d = kwargs.get('d', self.d)
        n = kwargs.get('n', self.n.copy())
        l = kwargs.get('l', self.l.copy())
        k = kwargs.get('k', self.k)

        return Grid(d, n, l, k)

    def comp(self, I=None, is_ind=False, is_inner=False):
        '''
        Compute grid points for given multi-indices.
        In case of the Chebyshev multidimensional grid points for every axis k
        are calculated as x = cos(i[k]*pi/(n[k]-1)), where n[k] is a total
        number of points for selected axis k, and then these points are scaled
        according to the grid limits l.

        INPUT:

        I - indices of grid points
        type1: None
        type2: int
        type3: list [number of points] of int
        type4: ndarray [number of points] of int
        type5: list [dimensions] of int
        type6: ndarray [dimensions] of int
        type7: list [dimensions, number of points] of int
        type8: ndarray [dimensions, number of points] of int
        * If is None (type1), then the full grid will be constructed.
        * Type2 is available only for the case of only one point in 1D.
        * Type3 and type4 are available only for 1D case. Type5 and type6
        * are available for only one point in the multidimensional case.

        is_ind - flag:
            True  - only prepared indices of points will be returned
            False - constructed spatial grid points will be returned
        type: bool

        is_inner - flag:
            True  - only inner (not on boundary) grid points will be constructed
            False - all grid points will be constructed
        type: bool

        OUTPUT:

        I - (if is_ind == True) prepared indices of grid points
        type: ndarray [dimensions, number of points] of int

        X - (if is_ind == False) calculated grid points
        type: ndarray [dimensions, number of points] of float

        TODO Maybe raise error for attempt of construction too large full grid.

        TODO Add check for I (>= 0 and < n).
        '''

        if I is None:
            I = []
            for i in range(self.d):
                I_ = np.arange(self.n[i])
                if is_inner:
                    I_ = I_[1:-1]
                I.append(I_.reshape(1, -1))
            I = np.meshgrid(*I, indexing='ij')
            I = np.array(I).reshape((self.d, -1), order='F')
        else:
            if isinstance(I, (int, float)):
                I = [int(I)]
            if isinstance(I, list):
                I = np.array(I, dtype='int')
            if not isinstance(I, np.ndarray):
                raise ValueError('Invalid grid points.')
            if len(I.shape) == 1 and self.d == 1:
                I = I.reshape(1, -1) # many one-dimensional points
            if len(I.shape) == 1 and self.d >= 2:
                I = I.reshape(-1, 1) # one multidimensional point
            if len(I.shape) != 2:
                raise ValueError('Invalid shape for grid points.')
            if I.shape[0] != self.d:
                raise ValueError('Invalid dimension for grid points.')

        if is_ind:
            return I.astype(int)

        n_ = np.repeat(self.n.reshape((-1, 1)), I.shape[1], axis=1)
        l1 = np.repeat(self.l[:, 0].reshape((-1, 1)), I.shape[1], axis=1)
        l2 = np.repeat(self.l[:, 1].reshape((-1, 1)), I.shape[1], axis=1)

        if self.k == 'u':
            t = I * 1. / (n_ - 1)
            X = t * (l2 - l1) + l1
        if self.k == 'c':
            t = np.cos(np.pi * I / (n_ - 1))
            X = t * (l2 - l1) / 2. + (l2 + l1) / 2.

        return X

    def find(self, x):
        '''
        Find the nearest flatten grid index for the given spatial point.

        INPUT:

        x - spatial point
        type1: float
        type2: list [dimensions] of float
        type3: ndarray [dimensions] of float
        * May be float (type1) for the 1-dimensional case.

        OUTPUT:

        i - flatten index of the grid point.
        type: int >= 0 and < prod(n)

        TODO add flag to select output (flatten or multi index).
        '''

        x = self._prep_poi(x)
        n = self.n
        l1 = self.l[:, 0]
        l2 = self.l[:, 1]

        if self.k == 'u':
            i = (x - l1) * (n - 1) / (l2 - l1)
        if self.k == 'c':
            t = (2. * x - l2 - l1) / (l2 - l1)
            t[t > +1.] = +1.
            t[t < -1.] = -1.
            i = np.arccos(t) * (n - 1) / np.pi

        i = np.rint(i).astype(int)
        i[i <= 0] = 0
        i[i >= n] = n[i >= n] - 1.

        return self.indf(i)

    def indm(self, i_f):
        '''
        Construct multi index from the given flatten index
        of the grid point.

        INPUT:

        i_f - flatten index of the grid point
        type:  int >= 0 and < prod(n)

        OUTPUT:

        i_m - multi index of the grid point
        type: ndarray [dimensions] of int >= 0, i_m[i] < n[i] for each i
        '''

        i_m = np.unravel_index(i_f, self.n, order='F')

        return i_m

    def indf(self, i_m):
        '''
        Construct flatten index from the given multi index
        of the grid point.

        INPUT:

        i_m - multi index of the grid point
        type1: list [dimensions] of int >= 0, i_m[i] < n[i] for each i
        type2: ndarray [dimensions] of int >= 0, i_m[i] < n[i] for each i

        OUTPUT:

        i_f - flatten index of the grid point.
        type: int >= 0 and < prod(n)
        '''

        i_f = np.ravel_multi_index(i_m, self.n, order='F')

        return i_f

    def rand(self, n):
        '''
        Generate random points inside the grid limits.
        * Uniform distribution is used.

        INPUT:

        n - total number of points
        type: int > 0

        OUTPUT:

        X - calculated random points
        type: ndarray [dimensions, n] of float

        TODO Add selection of the distribution kind.
        '''

        n = int(n)
        if n <= 0:
            raise ValueError('Invalid number of points.')

        l1 = np.repeat(self.l[:, 0].reshape((-1, 1)), n, axis=1)
        l2 = np.repeat(self.l[:, 1].reshape((-1, 1)), n, axis=1)

        return l1 + np.random.random((self.d, n)) * (l2 - l1)

    def info(self, is_ret=False):
        '''
        Present info about the grid.

        INPUT:

        is_ret - flag:
            True  - return string info
            False - print string info
        type: bool

        OUTPUT:

        s - (if is_ret) string with info
        type: str
        '''

        is_square = self.is_square()

        s = '------------------ Grid\n'

        k = '???'
        if self.k == 'u':
            k = 'Uniform'
        if self.k == 'c':
            k = 'Chebyshev'
        s+= 'Kind             : %s\n'%k

        s+= 'Dimensions       : %-2d\n'%self.d

        s+= '%s             : '%('Mean' if not is_square else '    ')
        s+= 'Poi %-3d | '%self.n0
        s+= 'Min %-6.3f | '%self.l1
        s+= 'Max %-6.3f |\n'%self.l2

        if not is_square:
            for i, [n, l] in enumerate(zip(self.n, self.l)):
                if i >= 5 and i < self.d - 5:
                    if i == 5:
                        s+= ' ...             : ...\n'
                    continue
                s+= 'Dim. # %-2d        : '%(i+1)
                s+= 'Poi %-3d | '%n
                s+= 'Min %-6.3f | '%l[0]
                s+= 'Max %-6.3f |\n'%l[1]

        if not s.endswith('\n'):
            s+= '\n'
        if is_ret:
            return s
        print(s[:-1])

    def plot(self, I=None, n=None, x0=None):
        '''
        Plot the full grid or some grid points or some random points.
        * Only 2-dimensional case is supported.

        I - indices of grid points for plot
        * See description in Grid.comp function.

        n - total number of random points
        * See description in Grid.rand function.
        * If is set, then random points are used.

        x0 - special spatial point for present on the plot if required
        type1: None
        type2: float
        type3: list [dimensions] of float
        type4: ndarray [dimensions] of float
        * May be float (type2) for the 1-dimensional case.
        '''

        if self.d != 2:
            raise NotImplementedError('Invalid dimension for plot.')

        if I is not None and n is not None:
            raise ValueError('Both I and n are set.')

        X = self.rand(n) if n is not None else self.comp(I)

        for k in range(X.shape[1]):
            x = X[:, k]
            plt.scatter(x[0], x[1])
            plt.text(x[0]+0.1, x[1]-0.1, '%d'%k)

        x0 = self._prep_poi(x0, is_opt=True)
        if x0 is not None:
            plt.scatter(
                x0[0], x0[1], s=80., c='#8b1d1d', marker='*', alpha=0.9
            )

        plt.show()

    def is_in(self, x, eps=1.E-20):
        '''
        Check if given point is inside the grid.

        INPUT:

        x - spatial point
        type1: float
        type2: list [dimensions] of float
        type3: ndarray [dimensions] of float
        * May be float (type1) for the 1-dimensional case.

        eps - accuracy for check
        type: float > 0

        OUTPUT:

        res - True if point is inside the grid and False otherwise
        type: bool
        '''

        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError('Invalid accuracy eps (should be > 0).')

        x = self._prep_poi(x)
        l = self.l

        return np.min(x - l[:, 0]) > eps and np.min(l[:, 1] - x) > eps

    def is_out(self, x, eps=1.E-20):
        '''
        Check if given point is out of the grid.

        INPUT:

        x - spatial point
        type1: float
        type2: list [dimensions] of float
        type3: ndarray [dimensions] of float
        * May be float (type1) for the 1-dimensional case.

        eps - accuracy for check
        type: float > 0

        OUTPUT:

        res - True if point is out of the grid and False otherwise
        type: bool
        '''

        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError('Invalid accuracy eps (should be > 0).')

        x = self._prep_poi(x)
        l = self.l

        return np.max(l[:, 0] - x) > eps or np.max(x - l[:, 1]) > eps

    def is_border(self, x, eps=1.E-20):
        '''
        Check if given point is lies on the grid border.

        INPUT:

        x - spatial point
        type1: float
        type2: list [dimensions] of float
        type3: ndarray [dimensions] of float
        * May be float (type1) for the 1-dimensional case.

        eps - accuracy for check
        type: float >= 0

        OUTPUT:

        res - True if point is on the grid border and False otherwise
        type: bool

        TODO Replace by more accurate check.
        '''

        if self.is_in(x, eps):
            return False
        if self.is_out(x, eps):
            return False
        return True

    def is_sym(self, eps=1.E-20):
        '''
        Check if grid is symmetric (l_min = - l_max for all dimensions).

        INPUT:

        eps - accuracy for check
        type: float > 0

        OUTPUT:

        res - True if grid is symmetric and False otherwise
        type: bool
        '''

        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError('Invalid accuracy eps (should be > 0).')

        return np.max(np.abs(self.l[:, 0] + self.l[:, 1])) < eps

    def is_square(self, eps=1.E-20):
        '''
        Check if grid is square (all dimensions are equal in terms of
        numbers of grid points and limits).

        INPUT:

        eps - accuracy for check
        type: float > 0

        OUTPUT:

        res - True if grid is square and False otherwise
        type: bool

        TODO Rename this function.
        '''

        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError('Invalid accuracy eps (should be > 0).')

        def is_non_zero(x):
            return np.max(np.abs(x)) > eps

        n0 = self.n0
        if is_non_zero(self.n - n0):
            return False

        l0 = np.array([self.l1, self.l2]).reshape((1, -1))
        l0 = np.repeat(l0, self.d, axis=0)
        if is_non_zero(self.l - l0):
            return False

        return True

    def _prep_poi(self, x=None, is_opt=False):
        '''
        Check and prepare given point according to:
        int, float, list [d], np.ndarray [d] => np.ndarray [d].

        INPUT:

        x - spatial point
        type1: None
        type2: float
        type3: list [dimensions] of float
        type4: ndarray [dimensions] of float
        * May be float (type2) for the 1-dimensional case.

        is_opt - flag:
            True  - if x is None, then function return None
            False - if x is None, then function raises error
        type: bool

        OUTPUT:

        res - prepared spatial point
        type: ndarray [dimensions] of float
        '''

        if x is None:
            if not is_opt:
                raise ValueError('The spatial point is not set.')
            return

        if isinstance(x, (int, float)):
            x = [float(x)]
        if isinstance(x, list):
            x = np.array(x)

        if not isinstance(x, np.ndarray) or len(x.shape) != 1:
            raise ValueError('Invalid type or shape of the spatial point.')
        if x.shape[0] != self.d:
            raise ValueError('Invalid dimension of the spatial point.')

        return x
