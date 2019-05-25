from attr import attrs, attrib
import numpy as np
from lmfit import minimize, Parameters
import plotly.graph_objs as go
from functools import partial
from IPython.display import Latex, display


@attrs
class Utility:
    """
    * Can be constructed from an already made function, so no points, no fitting, just plots
    and addition and multiplication methods. which is useful for when returning a MAUF
    where the function would be calculated and returned
    """

    name = attrib()
    function = attrib(default=None)

    @classmethod
    def from_function(cls, name, function):
        return cls(name=name, function=function)

    @classmethod
    def from_assessment(cls, name):
        return cls(name=name)

    def perfect_assess(self, vector, is_cost=False, method="least_squares"):
        # TODO: check if it is cost
        # TODO: check for when best and worst are not extrema
        worst = vector.min()
        best = vector.max()
        if is_cost:
            worst, best = best, worst
        params = Parameters()
        params.add("worst", value=worst, vary=False)
        params.add("lower_middle", min=vector.min(), max=vector.max())
        params.add("middle", min=vector.min(), max=vector.max())
        params.add("upper_middle", min=vector.min(), max=vector.max())
        params.add("best", value=best, vary=False)

        def objective(params):
            worst = params["worst"]
            lower_middle = params["lower_middle"]
            middle = params["middle"]
            upper_middle = params["upper_middle"]
            best = params["best"]
            u = Utility.from_assessment(name="u")
            x = np.array([worst, lower_middle, middle, upper_middle, best])
            u.assess(points=x)
            return u.fit().chisqr

        return minimize(objective, params, method=method)

    def assess(self, points=None, interactive=False):
        """Take in five points for direct assessment. When interactive
        is True, a Hint is displayed at each step of the assessment.

        Keyword Arguments:
            points {np.array | list-like} -- a list of five points (default: {None})
            interactive {bool} -- Toggle for interactively setting the points (default: {False})
        """
        # TODO: add a way to choose points optimally based
        # on a vector to fit a model by minimizing a chi-square
        self.points = points
        return self.points

    @property
    def points(self):
        return np.array(self._points)

    @points.setter
    def points(self, points_array):
        # TODO: verify length is 5
        self._points = points_array

    def fit(self):
        self.params = Parameters()
        self.params.add("a", value=1.0)
        self.params.add("b", value=1.0)
        self.params.add("c", value=1.0)

        self.result = minimize(
            self.residuals, self.params, args=(self.points, np.linspace(0, 1, 5))
        )
        return self.result

    def model(self, x, params=None):
        """
        Return an array of y values based on function calculated inside
        """
        if params is None:
            params = self.result.params
        a = params["a"]
        b = params["b"]
        c = params["c"]

        return a + b ** (-c * x)

    def residuals(self, params, x, data):
        """
        Return the gap
        """
        a = params["a"]
        b = params["b"]
        c = params["c"]

        return self.model(x, params) - data

    def plot(self):
        trace1 = go.Scatter(x=self.points, y=np.linspace(0, 1, 5))
        x = np.linspace(self.points.min(), self.points.max(), 50)
        y = self.model(x=x)
        trace2 = go.Scatter(x=x, y=y)
        return [trace1, trace2]

    def __mul__(self, other):
        # u(x, y) = kxux(x)+ kyuy(y)+kxyux(x)uy(y)
        return lambda x, y: self.model(x) + other.model(y)
