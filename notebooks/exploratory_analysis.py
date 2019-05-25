# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.2
#   kernelspec:
#     display_name: Python [conda env:utility_theory] *
#     language: python
#     name: conda-env-utility_theory-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from itertools import combinations

# %% [markdown]
# ## Ideas
# * minimize chi-square

# %%
import numpy as np
import pandas as pd
import qgrid
import attr
from plotly.offline import iplot
import plotly.graph_objs as go
from lmfit import minimize, Parameters

from src.data.loader import load_data
from src.data.utils import get_indicator
from src.utility import Utility
from IPython.display import Latex, display
from lmfit import Parameters, minimize

# qgrid.enable()

# %%
data = load_data()

# %%
data.columns

# %%
name = "Number of free practice Doctors"
nursing_staff = Utility.from_assessment(name=name)
perfect_fit = nursing_staff.perfect_assess(data.loc[2000, name].values, is_cost=True)

# %%
nursing_staff.assess(points=list(perfect_fit.params.valuesdict().values()))
res = nursing_staff.fit()
iplot(nursing_staff.plot())

# %%
nursing_staff = Utility.from_assessment(name=name)
perfect_fit = nursing_staff.perfect_assess(data.loc[2000, name].values, is_cost=True)

# %%
res

# %%
nursing_staff.assess(points=list(perfect_fit.params.valuesdict().values()))
res = nursing_staff.fit()
iplot(nursing_staff.plot())

# %% [markdown]
# > *As discussed in Pratt [1964], special risk attitudes restrict the functional form of single-attribute utility functions. A common utility function is the exponential utility function $$u(x) = a + b^{-cx}$$ where $a, b > 0, c > 0$ are scaling constants.*
#
# [Keeney, R. (1982). Decision Analysis: An Overview. Operations Research, 30(5), 803-838.](https://sci-hub.tw/10.2307/170347)
#

# %% [markdown]
# ## Evaluating scaling constants

# %% [markdown]
# ## References
#
# 1. [Application of Multi-Attribute Utility Theory to Measure Social Preferences for Health States. Operations Research, 30(6), 1043–1069.](https://sci-hub.tw/10.1287/opre.30.6.1043)
# 1. [Multiattribute and Single-Attribute Utility Functions for the Health Utilities Index Mark 3 System. Medical Care,Vol. 40, No. 2 (Feb., 2002), pp. 113-128](https://sci-hub.tw/10.2307/3767552)
# 1. [Contractor selection using multicriteria utility theory: An additive model. Building and Environment, 33(2-3), 105–115.](https://sci-hub.tw/10.1016/S0360-1323(97)00016-4)
# 1. [Keeney, R. (1982). Decision Analysis: An Overview. Operations Research, 30(5), 803-838.](https://sci-hub.tw/10.2307/170347)
# 1. [The Multi-attribute Utility Method. The Measurement and Analysis of Housing Preference and Choice, 101–125.](https://sci-hub.tw/10.1007/978-90-481-8894-9_5)
# 1. (https://sci-hub.tw/10.1007/0-387-23081-5_7)
