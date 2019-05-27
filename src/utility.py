import attr
import numpy as np
from lmfit import minimize, Parameters, Parameter
import plotly.graph_objs as go
import diofant
from functools import reduce
from operator import mul


@attr.s
class Utility(object):
    name = attr.ib()
    points = attr.ib(default=None)
    optimal_fit = attr.ib(default=False)
    data = attr.ib(default=None)
    is_cost = attr.ib(default=False)
    method = attr.ib(default="least_squares")
    # Useful fo when adding an already calculated function
    params = attr.ib(default=None)
    # For MAUFs
    saufs = attr.ib(default={})

    @property
    def best(self):
        if self.saufs:
            # MAUF
            self.__best = {}
            for key, value in self.saufs.items():
                self.__best[key] = value.best
        else:
            # SAUF
            if self.optimal_fit:
                self.__best = self.data.max()
                if self.is_cost:
                    self.__best = self.data.min()
            else:
                self.__best = self.points.max()
                if self.is_cost:
                    self.__best = self.min.min()
        return self.__best

    @property
    def worst(self):
        if self.saufs:
            # MAUF
            self.__worst = {}
            for key, value in self.saufs.items():
                self.__worst[key] = value.worst
        else:
            # SAUF
            if self.optimal_fit:
                self.__worst = self.data.min()
                if self.is_cost:
                    self.__worst = self.data.max()
            else:
                self.__worst = self.points.min()
                if self.is_cost:
                    self.__worst = self.min.max()
        return self.__worst

    def assess(self):
        # TODO: mutually exclusive args
        # TODO: check for when best and worst are not extrema

        if self.optimal_fit:
            points_params = Parameters()
            worst = Parameter("worst", value=self.worst, vary=False)
            lower_middle = Parameter("lower_middle")
            middle = Parameter("middle")
            upper_middle = Parameter("upper_middle")
            best = Parameter("best", value=self.best, vary=False)
            for param in [lower_middle, middle, upper_middle]:
                param.set(min=self.data.min(), max=self.data.max())
            points_params.add_many(worst, lower_middle, middle, upper_middle, best)

            def objective(params):
                v = params.valuesdict()
                u = Utility(name="u", points=list(v.values()))
                u.fit()
                chisqr = u.result.chisqr
                return chisqr

            self.optimal_points = minimize(objective, points_params, method=self.method)
            v = self.optimal_points.params.valuesdict()
            self.points = np.array(list(v.values()))
        return self.points

    def fit(self):

        if not self.params:
            self.params = Parameters()
            a = Parameter(name="a", value=1)
            b = Parameter(name="b", value=1)
            c = Parameter(name="c", value=1)
            self.params.add_many(a, b, c)

        self.result = minimize(
            self.residuals,
            self.params,
            args=(self.points, np.linspace(0, 1, 5)),
            nan_policy="propagate",
        )
        return self.result

    def model(self, x, params=None):
        """
        Return an array of y values based on function calculated inside
        """
        if params is None:
            params = self.result.params

        v = params.valuesdict()

        y = v["a"] + v["b"] ** (-v["c"] * np.array(x))
        return y

    def normalized_model(self, x, params=None):
        """
        Return an array of y values based on function calculated inside
        Used only after optimality
        """
        if params is None:
            params = self.result.params
        v = params.valuesdict()

        lower = self.model(self.worst)
        upper = self.model(self.best)
        y = v["a"] + v["b"] ** (-v["c"] * np.array(x))
        return (y - lower) / (upper - lower)

    def residuals(self, params, x, data):
        """
        Return the gap
        """
        return self.model(x, params=params) - data

    def plot(self):
        trace1 = go.Scatter(x=self.points, y=np.linspace(0, 1, 5))
        x = np.linspace(self.points.min(), self.points.max(), 50)
        y = self.normalized_model(x=x)
        trace2 = go.Scatter(x=x, y=y)
        return [trace1, trace2]

    def scale(self, **kwargs):
        # TODO: ensure provided kwargs are keys in saufs
        # pass the value of the scaling constant with key its name
        self.scaling_constants = kwargs
        ## TODO: Assuming order, should find a way to pair each ki to its value by name
        ## may not be necessary since we do not care about the value of each UF
        scaling_constants = np.array(list(self.scaling_constants.values()))
        if sum(scaling_constants) == 1:
            self.k = 0
        k = diofant.symbols("k")
        rhs = reduce(mul, k * scaling_constants + 1, 1)
        lhs = k + 1
        sols = diofant.solve(lhs - rhs)
        # as opposed to complex, filter real solutions
        real_solutions = filter(
            lambda sol: -1 <= sol.get(k) <= 1,
            filter(lambda sol: sol.get(k).is_Float, sols),
        )
        not_null = [sol.get(k) for sol in real_solutions if sol.get(k) != 0]
        if len(not_null) == 0:
            self.k = 0
        else:
            self.k = not_null[0]
        return self.k

    def eval(self, **kwargs):
        if self.saufs:
            lhs = 1 + self.k
        else:
            return self.normalized_model(kwargs["x"])

    def expr(self):
        pass

    def __mul__(self, other):
        saufs = {self.name: self, other.name: other}
        mauf = Utility(name="_".join(saufs.keys()), saufs=saufs)
        return mauf

    def __rmul__(self, other):
        return self.__mul__(other)
