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
import numpy as np
from plotly.offline import iplot
import plotly.io as pio

from src.data.loader import load_data
from src.utility import Utility
import diofant
from pprint import pprint as print
import cufflinks as cf
cf.go_offline()

# %%
data = load_data()

# %%
data.set_index("governorate")[
    [
        "Number of institutes, centers and specialized hospitals in public sector",
        "Number of public district hospitals",
        "Number of regional public hospitals",
        "The number of public hospitals",
    ]
].fillna(0)

# %% [markdown]
# ## Indicators

# %% [markdown]
# 1. Access to professionals
#     * Number of doctors / 1000 inhabitants
#     * Number of paramedical staff / 1000 inhabitants
#
# 1. Access to facilities
#     * Number of public basic health centers / 1000 inhabitants
#     * Number of beds / 1000 inhabitants
#
# 1. Access to medicine
#     * Number of pharmacies / 1000 inhabitants

# %% [markdown] {"toc-hr-collapsed": false}
# ## Access to professionals

# %% [markdown]
# ### Number of doctors / 1000 inhabitants

# %%
u_ndocs = Utility(name="ndocs", optimal_fit=True, data=data["Number of doctors"].values)
u_ndocs.assess()
u_ndocs.fit();

# %%
data.columns

# %%
u_ndocs.result.params.valuesdict()

# %% [markdown]
# ### Number of paramedical staff / 1000 inhabitants

# %%
u_nparam = Utility(name="nparam", optimal_fit=True, data=data["Number of paramedical staff"].values)
u_nparam.assess()
u_nparam.fit();

# %%
u_nparam.result.params.valuesdict()

# %% [markdown]
# ### Access to professionals MAUF

# %%
access_to_pros = u_ndocs * u_nparam
access_to_pros.name = "AP"

# %%
access_to_pros.scale(ndocs=0.7, nparams=0.6)

# %% [markdown] {"toc-hr-collapsed": false}
# ## Access to facilities

# %% [markdown]
# ### Number of public basic health centers / 1000 inhabitants

# %%
u_ncenters = Utility(name="ncenters", optimal_fit=True, data=data["Number of public basic health centers"].values)
u_ncenters.assess()
u_ncenters.fit();

# %%
u_ncenters.result.params.valuesdict()

# %% [markdown]
# ### Number of beds in the public health sector / 1000 inhabitants

# %%
u_nbeds = Utility(name="nbeds", optimal_fit=True, data=data["Number of beds in the public health sector"].values)
u_nbeds.assess()
u_nbeds.fit();

# %%
u_nbeds.result.params.valuesdict()

# %% [markdown]
# ### Access to facilities MAUF

# %%
access_to_facilities = u_ncenters * u_nbeds
access_to_facilities.name = "AF"

# %%
access_to_facilities.scale(u_ncenters=0.8, u_nbeds=0.4)

# %% [markdown] {"toc-hr-collapsed": false}
# ## Access to medicine

# %% [markdown] {"toc-hr-collapsed": false}
# ### Number of pharmacies / 1000 inhabitants

# %%
u_npharma = Utility(name="npharma", optimal_fit=True, data=data["Number of pharmacies"].values)
u_npharma.assess()
u_npharma.fit();

# %% [markdown]
# ### Access to medicine MAUF

# %%
access_to_medicine = u_npharma
access_to_medicine.name = "AM"

# %% [markdown]
# ## MAUF

# %%
mauf = access_to_facilities * access_to_pros * access_to_medicine

# %%
mauf.scale(ncenters_nbeds_ndocs_nparam=0.7, npharma=0.2)

# %%
kwargs = {"ncenters_nbeds_ndocs_nparam": 0.7, "npharma": 0.5}

# %%
mauf.saufs.keys()

# %%
X = diofant.symbols("X")
diofant.solve(X + 1 - (0.8 * X + 1) * (0.6 * X + 1) * (0.7 * X + 1))

# %% [markdown]
# ## Figures

# %% [markdown]
# #### Number of doctors

# %%
fig = data.set_index("governorate")["Number of doctors"].apply(
    u_ndocs.normalized_model
).sort_values().iplot(
    kind="bar",
    title="Ranking of governorates based on the utility of number of doctors per 1000 inhabitants",
    asFigure=True
)

# %%
pio.write_image(fig, 'Figures/ND.svg')

# %% [markdown]
# #### Number of paramedical staff

# %%
fig = data.set_index("governorate")["Number of paramedical staff"].apply(
    u_nparam.normalized_model
).sort_values().iplot(
    kind="bar",
    title="Ranking of governorates based on the utility of number of paramedical staff per 1000 inhabitants",
    asFigure=True
)

# %%
pio.write_image(fig, 'Figures/NP.svg')

# %% [markdown]
# #### Number of basic health centers

# %%
fig = data.set_index("governorate")["Number of public basic health centers"].apply(
    u_ncenters.normalized_model
).sort_values().iplot(
    kind="bar",
    title="Ranking of governorates based on the utility of Number of public basic health centers staff per 1000 inhabitants",
    asFigure=True
)

# %%
pio.write_image(fig, 'Figures/PH.svg')

# %% [markdown]
# #### Number of beds

# %%
fig = data.set_index("governorate")["Number of beds in the public health sector"].apply(
    u_nbeds.normalized_model
).sort_values().iplot(
    kind="bar",
    title="Ranking of governorates based on the utility of Number of beds in the public health sector per 1000 inhabitants",
    asFigure=True
)

# %%
pio.write_image(fig, 'Figures/NB.svg')

# %% [markdown]
# #### Number of pharmacies

# %%
fig = data.set_index("governorate")["Number of pharmacies"].apply(
    u_npharma.normalized_model
).sort_values().iplot(
    kind="bar",
    title="Ranking of governorates based on the utility of Number of pharmacies 1000 inhabitants",
    asFigure=True
)

# %%
pio.write_image(fig, 'Figures/NPH.svg')


# %% [markdown]
# ### MUAF calc

# %% [markdown]
# ### Access to professionals

# %%
def UAP(ND, NP):
    kap = -0.71428
    scaled_ND = 0.7 * kap * u_ndocs.normalized_model(ND) + 1
    scaled_NP = 0.6 * kap * u_nparam.normalized_model(NP) + 1
    u = ((scaled_ND * scaled_NP) -1) / kap
    return u

def normalized_UAP(ND, NP):
    lower = UAP(u_ndocs.worst, u_nparam.worst)
    upper = UAP(u_ndocs.best, u_nparam.best)
    u = UAP(ND, NP)
    return (u - lower) / (upper - lower)


# %%
normalized_UAP(u_ndocs.worst, u_nparam.worst)

# %% [markdown]
# can be explained by the fact that these regions are close to Tunis or other major cities, thus having a lower value for the index inside the governorate may not reflect the reality of the health situation for ppl as it does not account for closeness to other cities, moreover, this does not account for experience of staff, quality of service, and may actually reflect a state policy of forcing new graduates to interior regions

# %%
data.set_index("governorate").pipe(
    lambda df: df[["Number of doctors", "Number of paramedical staff"]]
).pipe(
    lambda df: df.assign(
        UAP=normalized_UAP(df["Number of doctors"], df["Number of paramedical staff"])
    )
).sort_values(
    "UAP"
)#.iplot(
 #   kind="bar",
 #   y="UAP",
 #   title="Ranking of governorates based on Access to Professionals",
 #   asFigure=True,
#)

# %%
pio.write_image(fig, 'Figures/AP.svg')


# %% [markdown]
# ### Access to facilities

# %%
def UAF(PH, NB):
    kaf = access_to_facilities.k
    scaled_PH = 0.8 * kaf * u_ncenters.normalized_model(PH) + 1
    scaled_NB = 0.4 * kaf * u_nbeds.normalized_model(NB) + 1
    u = ((scaled_PH * scaled_NB) - 1) / kaf
    return u


def normalized_UAF(PH, NB):
    lower = UAF(u_ncenters.worst, u_nbeds.worst)
    upper = UAF(u_ncenters.best, u_nbeds.best)
    u = UAF(PH, NB)
    return (u - lower) / (upper - lower)


# %%
normalized_UAF(u_ncenters.best, u_nbeds.best)

# %%
data.set_index("governorate").pipe(
    lambda df: df.assign(
        UAF=normalized_UAF(
            df["Number of public basic health centers"],
            df["Number of beds in the public health sector"],
        )
    )
).sort_values(
    "UAF"
)[["Number of public basic health centers","Number of beds in the public health sector", "UAF" ]]

# %% [markdown]
# ### Access to medicine

# %%
data.set_index("governorate").pipe(lambda df: df.loc[:, ["Number of pharmacies"]]).pipe(
    lambda df: df.assign(
        UAM=df["Number of pharmacies"].apply(u_npharma.normalized_model)
    )
).sort_values("UAM")#.iplot(kind="bar", y="UAM")


# %% [markdown]
# ### Final MAUF

# %%
def MAUF(ND, NP, PH, NB, NPH):
    kmauf = -0.969928
    scaled_AP = 0.8 * kmauf * normalized_UAP(ND, NP) + 1
    scaled_AF = 0.6 * kmauf * normalized_UAF(PH, NB) + 1
    scaled_AM = 0.7 * kmauf * u_npharma.normalized_model(NPH) + 1

    u = ((scaled_AP * scaled_AF * scaled_AM) - 1) / kmauf
    return u


def normalized_MAUF(ND, NP, PH, NB, NPH):
    uts = [u_ndocs, u_nparam, u_ncenters, u_nbeds, u_npharma]
    worst = [ut.worst for ut in uts]
    best = [ut.best for ut in uts]

    lower = MAUF(*worst)
    upper = MAUF(*best)
    u = MAUF(ND, NP, PH, NB, NPH)
    return (u - lower) / (upper - lower)


# %%
uts = [u_ndocs, u_nparam, u_ncenters, u_nbeds, u_npharma]
worst = [ut.worst for ut in uts]
best = [ut.best for ut in uts]

# %%
worst

# %%
normalized_MAUF(*best)

# %%
data.columns

# %%
data.set_index("governorate").pipe(
    lambda df: df.assign(
        MAUF=normalized_MAUF(
            df["Number of doctors"],
            df["Number of paramedical staff"],
            df["Number of public basic health centers"],
            df["Number of beds in the public health sector"],
            df["Number of pharmacies"]
        )
    )
).sort_values("MAUF").drop(columns=["Population"]).iplot()

# %%
