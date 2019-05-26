import attr
import numpy as np
from lmfit import minimize, Parameters, Parameter
import plotly.graph_objs as go
import diofant


@attr.s
class Utility(object):
    name = attr.ib()
    points = attr.ib(default=None)
    optimal_fit = attr.ib(default=False)
    data = attr.ib(default=None)
    is_cost = attr.ib(default=False)
    method = attr.ib(default="least_squares")
    # Usefule fo when adding an already calculated function
    params = attr.ib(default=None)

    def assess(self):
        # TODO: mutually exclusive args
        # TODO: check for when best and worst are not extrema

        if self.optimal_fit:
            self.worst = self.data.min()
            self.best = self.data.max()

            if self.is_cost:
                self.worst, self.best = self.best, self.worst
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
        else:

            # TODO: DRY this comparison
            self.worst = self.points.min()
            self.best = self.points.max()

            if self.is_cost:
                self.worst, self.best = self.best, self.worst
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

    def __mul__(self, other):
        x, y = diofant.symbols("x, y")
        f = lambda x: x.replace(" ", "_").lower()
        kx, ky = diofant.symbols(r"k_{}, k_{}".format(f(self.name), f(other.name)))
        sauf_x = lambda x: self.normalized_model(x)
        sauf_y = lambda y: other.normalized_model(y)
        mauf = kx * sauf_x(x) + ky * sauf_y(y) + (1 - kx - ky) * sauf_x(x) * sauf_y(y)

        return mauf
