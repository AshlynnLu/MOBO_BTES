from __future__ import annotations

import pandas as pd

import torch
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)

from torch import Tensor

import numpy as np
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement

from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from botorch.models.transforms import Standardize

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective

import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from botorch.sampling.normal import IIDNormalSampler
from botorch.sampling.normal import NormalMCSampler

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize parameters
system_Ts_h = 40               # hot supply temperature (C)
multiplier_cooling = 1  # 1:HeatingDominated, 3:BalancedLoads
with_solar = True       # either with or without solar collectors

# COP^-1 coefficients for BTES heat pump
if system_Ts_h==65:
    ad = -0.005915
    bd = 0.3844
else:  
    ad = -0.006387
    bd = 0.2555


# Load the Excel file
heatDemand = pd.read_excel('/home/icnfs/ma/k/kl1323/myfolder/heating_load.xlsx')
heatDemand = heatDemand['Total '].values

coolDemand = 0

netEnergy = heatDemand - coolDemand

weather = pd.read_excel('/home/icnfs/ma/k/kl1323/myfolder/weather_conditions.xlsx', header=1)
irradiation = weather['Solar radiation (W/m2)'].values # W/m2
Taa = weather['Temp(°C)'].values

carbon = pd.read_csv("/home/icnfs/ma/k/kl1323/myfolder/df_fuel_ckan.csv")
# Convert the 'DATETIME' column to datetime if it's not already
carbon['DATETIME'] = pd.to_datetime(carbon['DATETIME'])
# Filter the DataFrame for the year 2019
carbon = carbon[carbon['DATETIME'].dt.year == 2019]
# Set the 'DATETIME' column as the index of the DataFrame
carbon.set_index('DATETIME', inplace=True)
# Resample the data to hourly, and take the average of each hour
carbon_hourly = carbon.resample('H').mean()
CO2 = carbon_hourly['CARBON_INTENSITY'].values # g/kWh

# Reshape data to new time resolution
dt = 24  # Time resolution in hours
Horizon = 8760 // dt  # Optimization horizon (hours)
data = np.vstack([netEnergy, irradiation / 1000, CO2, Taa]).T #in kW, kW/m^2, gCO2/kWh, degC
data_r = data.reshape(Horizon, dt, 4).mean(axis=1)

# Extract reshaped data
heat_load_np = data_r[0:7, 0]        # kW
# cool_load_np = -data_r[:, 0]       # kW
# Set negative demand to zero
# cool_load_np[cool_load_np < 0] = 0  
heat_load_np[heat_load_np < 0] = 0

pv_irradiation_np = data_r[0:7, 1]   # kW/m^2
pv_irradiation = torch.from_numpy(pv_irradiation_np).to(dtype=torch.float32, device="cpu")  # Specify the device as needed

CO2_intensity = data_r[0:7, 2] / 1e6  # Scale CO2 intensity to tons
CO2_intensity = torch.from_numpy(CO2_intensity).to(dtype=torch.float32, device="cpu")  # Specify the device as needed

Ta = data_r[0:7, 3]               # degreeC
# cool_load_np *= multiplier_cooling # Adjust cooling demand profile
# cool_load = torch.from_numpy(cool_load_np).to(dtype=torch.float32)
heat_load = torch.from_numpy(heat_load_np).to(dtype=torch.float32)

Horizon = 7

# Constants for temperature calculations
system_Ts_c = 6     # cold supply (C)
system_Ts_ex = 35   # exhaust temperature (reference) (C)
DTnet = 15          # network temp diff (T_d,hs - T_d,hr) (C)
# Temperature difference between the HTF and the air, constant
DThp_a = 10         # K
DTch_a = 10         # K

# Calculate COP for exhaust and base heating
COPh_ex_a = 0.5*((system_Ts_c+273.15)/(Ta+273.15+DTch_a-(system_Ts_c+273.15)))      # exhaust COP (air) -- 10K heat exchanger
COPh_heat_base_a = 0.5*(system_Ts_h+273.15)/(system_Ts_h+273.15-(Ta+273.15-DThp_a)) # base heating COP (air) -- 10K heat exchanger

# Initialize arrays for energy base and discharge activity
energy_base = np.zeros(Horizon)
disch_act = np.full(Horizon, -1) # Default = inactive

# Create reduced CO2 intensity profile
#CO2_intensity_a = CO2_intensity.copy()

# Adjust energy base and COP based on load conditions
# for i in range(Horizon):
#     if cool_load[i] > 0: # if cooling is needed
#         energy_base[i] = cool_load[i] / COPh_ex_a[i]
#         COPh_heat_base_a[i] = 10  # Set COP manually for some condition
#         # Uncomment the next line if you want to modify CO2 intensity
#         # CO2_intensity_a[i] = CO2_intensity[i] / 3
#     else: # if heating is needed
#         energy_base[i] = heat_load[i] / COPh_heat_base_a[i]
#         COPh_ex_a[i] = 10  # Set COP manually for another condition
#         disch_act[i] = 1  # Indicate discharge activity
for i in range(Horizon):
    energy_base[i] = heat_load[i] / COPh_heat_base_a[i]
    COPh_ex_a[i] = 10  # Set COP manually for another condition
    disch_act[i] = 1  # Indicate discharge activity

# Define the lifetime and cost parameters
inv_years_BTES = 60                         # lifetime of BTES (years)
inv_years = 20                              # lifetime of other equipment (years)
elec_cost = 22.68 / 1000                   # price of electricity (kEUR/kWh)
CO2_cost_a = np.arange(0.05, 0.45, 0.1)     # price of CO2 emissions (kEUR/tCO2) 
CO2_cost = 0.037

# Capital costs converted to annual costs
hp_capital = 0.577 / inv_years              # capital cost BTES heat pump (kEUR/kW)
# chill_capital = 0.577 / inv_years           # capital cost BTES chiller (kEUR/kW)
# chillg_capital = 0.577 / inv_years          # capital cost air heat pump (kEUR/kW)
hpg_capital = 0.577 / inv_years             # capital cost air chiller (kEUR/kW)
sol_capital = 0.500 / inv_years             # capital cost solar collectors (EUR/m^2) --- 500 EUR/m2
sol_tank_capital = 9 / (1000 * inv_years)   # capital cost solar tank (EUR/m^2) --- 500 EUR/m2

# solar panels and solar capacity constraints
sol_cap_max = 100                           # capacity limit (m^2)
sol_eff = 0.65                              # efficiency of solar panels
max_sol_gen = sol_eff * np.max(pv_irradiation_np) * dt / 2  # half of max solar generation (kWh/m^2)

# System constraints
T_storage_min = 6                   # BTES min temperature (degC)    
T_storage_max = 65                  # BTES max temperature (degC)
T_init_min = 6                      # BTES min initial temperature (degC)
T_init_max = 30                     # BTES max initial temperature (degC)

# Capacity limits based on loads
hp_cap_max = np.max(heat_load_np)      # BTES heat pump capacity limit (kW)
chill_cap_max = 0   # BTES chiller capacity limit (kW)
hpg_cap_max = np.max(heat_load_np)     # Air heat pump capacity limit in kW
chillg_cap_max = 0  # Air chiller capacity limit in kW

# Efficiency calculations
hpg_eff = 1.0 / COPh_heat_base_a    # 1/COP for air heat pump
chillg_eff = 1.0 / COPh_ex_a        # 1/COP for air chiller

# Some initialisations?
x0 = 0

DTc = 10
DTd = 10

# Constants
Cost_m = 66     # cost for drilling + materials (EURO/m)
k_gr = 2.4      # ground conductivity (W/mK)
h = 21.2        # dimensionless coefficient for ground losses at steady state
D_base = 51     # base diameter for calculations of BTES paramentes (identified UA from TRNSYS) (m)
base_UA = 136   # base UA value (kW/K)
V_base = np.pi * (D_base / 2)**2 * D_base  # base volume (m^3)
n_base = 144    # base number of boreholes

d = 31
M = 2

max_heat_load = max(heat_load)

class PROBLEM(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    """The constrained SRN problem.

    See [GarridoMerchan2020]_ for more details on this problem. Note that this is a
    minimization problem.
    """

    dim = 31
    num_objectives = 2
    num_constraints = 51
    
    # Compute the array for the upper bounds
    upper_bounds_array = sol_eff * pv_irradiation * 100 * sol_cap_max

    _bounds = ([(0.0, upper_bound) for upper_bound in upper_bounds_array] +  # sol_used
               [(0.0, sol_cap_max)] +  # sol_cap
               [(T_init_min, T_init_max)] +  # T_store[0]
               [(T_storage_min, T_storage_max)] * Horizon +  # T_store[1:365]
               [(0.0, hpg_cap_max)] * Horizon +  # hpg_out 
               [(0.0, hpg_cap_max)] +  # capacity
               [(0.0, 1.0)] * Horizon  # dm
               )
    
    _ref_point = [0.0, 0.0]  # TODO: Determine proper reference point

    # def __init__(self, CO2_intensity, negate: bool = False):
    #     super().__init__(negate=negate)
    #     self.CO2_intensity = CO2_intensity


    def evaluate_true(self, X: Tensor) -> Tensor:
        # sol_used = X[..., 0:Horizon]
        sol_cap = X[..., Horizon]
        # T_store = X[..., (Horizon+1):(Horizon+2)]
        hpg_out = X[..., (Horizon+2):(2*Horizon+2)]
        hpg_cap = X[..., 2*Horizon+2]
        # dm = X[..., (2*Horizon+3):].round() # binary?
        
        sol_inv = sol_cap * 100 * sol_capital + sol_cap * 100 * max_sol_gen * sol_tank_capital
        hpg_in = hpg_out * hpg_eff
        hpg_cost = hpg_cap * hpg_capital

        # Capital cost
        f1 = sol_inv + hpg_cost
        print("f1", f1)
        # Electrical cost
        f2 = torch.sum(hpg_in, dim=-1) * dt * elec_cost
        print("f2", f2)
        # CO2 emission cost
        f3 = torch.sum(CO2_intensity * (hpg_in), dim=-1) * dt * CO2_cost
        print("f3", f3)
        return torch.stack([f1 + f2, f3], dim=-1)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        sol_used = X[..., 0:Horizon]
        sol_cap = X[..., Horizon]
        T_store = X[..., (Horizon+1):(Horizon+2)]
        hpg_out = X[..., (Horizon+2):(2*Horizon+2)]
        hpg_cap = X[..., 2*Horizon+2]
        dm = X[..., (2*Horizon+3):].round() # binary?
        
        # Constraints (g>=0)
        eps = 1
        # solar
        sol_out = sol_eff * pv_irradiation * 100 * sol_cap.unsqueeze(1)
        sol_g1 = sol_eff * pv_irradiation * 100 * sol_cap_max - sol_out
        
        sol_inv = sol_cap * 100 * sol_capital + sol_cap * 100 * max_sol_gen * sol_tank_capital
        sol_g2 = sol_cap_max * 100 * sol_capital + sol_cap_max * 100 * max_sol_gen * sol_tank_capital - sol_inv
        sol_g2 = sol_g2.unsqueeze(1)
        sol_g3 = sol_out - (sol_used)
        del sol_out, sol_inv
        
        # thermal storage
        T_store[0] = 12
        T_store[..., -1] = T_store[..., 0]

        # Air hp
        hpg_in = hpg_out * hpg_eff
        hpg_cap_max_tensor = torch.tensor(hpg_cap_max, dtype=torch.float32)
        hpg_g1 = torch.min(hpg_cap, hpg_cap_max_tensor).unsqueeze(-1).expand(-1, Horizon) - hpg_in
        hpg_cost = hpg_cap * hpg_capital
        hpg_g2 = hpg_cap_max * hpg_capital - hpg_cost
        hpg_g2 = hpg_g2.unsqueeze(1)
        hpg_g3 = hpg_cap.unsqueeze(-1) - hpg_out
        del hpg_cap_max_tensor, hpg_cost, hpg_in

        # Energy balance heating
        heat_h1 = hpg_out + sol_used - heat_load - eps
        heat_h2 = heat_load - (hpg_out + sol_used) - eps
        del hpg_out, sol_used
        
        condition_mask2 = (T_store <= system_Ts_h)[..., 0:Horizon]
        dm_g1 = torch.where(condition_mask2, torch.tensor(1, dtype=dm.dtype), dm)
        del condition_mask2

        # Storage dynamics constraints
        T_store[..., 1:] = T_store[..., :-1]

        return torch.cat([sol_g1, sol_g2, sol_g3, 
                          hpg_g1, hpg_g2, hpg_g3, heat_h1, heat_h2, 
                          dm_g1], dim=-1)

problem = PROBLEM(negate=True).to(device)

def generate_initial_data(n):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj = problem(train_x.cpu()).to(device)
    train_x = train_x.to(device)
    # negative values imply feasibility in botorch
    train_con = -problem.evaluate_slack(train_x.cpu()).to(device)
    return train_x, train_obj, train_con

def initialize_model(train_x, train_obj, train_con):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    train_y = torch.cat([train_obj, train_con], dim=-1)
    models = []
    for i in range(train_y.shape[-1]):
        models.append(
            SingleTaskGP(
                train_x, train_y[..., i : i + 1], outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    model = model.to(device)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

BATCH_SIZE = 2
NUM_RESTARTS = 10
RAW_SAMPLES = 512

standard_bounds = torch.zeros(2, problem.dim)
standard_bounds[1] = 1

def optimize_qnehvi_and_get_observation(model, train_x, train_obj, train_con, sampler):
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""
    train_x = normalize(train_x, problem.bounds)
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        # specify that the constraint is on the last outcome
        constraints=[lambda Z: Z[..., -i] for i in range(1, 102)],
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    # negative values imply feasibility in botorch
    new_con = -problem.evaluate_slack(new_x)
    return new_x, new_obj, new_con

N_BATCH = 100
MC_SAMPLES = 256

hv = Hypervolume(ref_point=problem.ref_point)
hvs_qnehvi = []

# call helper functions to generate initial training data and initialize model
train_x_qnehvi, train_obj_qnehvi, train_con_qnehvi = generate_initial_data(
    n=2 * (d + 1)
)

mll_qnehvi, model_qnehvi = initialize_model(
    train_x_qnehvi, train_obj_qnehvi, train_con_qnehvi
)

# compute pareto front
is_feas = (train_con_qnehvi <= 0).all(dim=-1)
feas_train_obj = train_obj_qnehvi[is_feas]
if feas_train_obj.shape[0] > 0:
    pareto_mask = is_non_dominated(feas_train_obj)
    pareto_y = feas_train_obj[pareto_mask]
    # compute hypervolume
    volume = hv.compute(torch.log(pareto_y))
else:
    volume = 0.0

hvs_qnehvi.append(volume)

# run N_BATCH rounds of BayesOpt after the initial random batch
for iteration in range(1, N_BATCH + 1):
    # fit the models
    fit_gpytorch_mll(mll_qnehvi)
    
    # define the qParEGO and qNEHVI acquisition modules using a QMC sampler
    # qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
    qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # optimize acquisition functions and get new observations
    new_x_qnehvi, new_obj_qnehvi, new_con_qnehvi = optimize_qnehvi_and_get_observation(
        model_qnehvi, train_x_qnehvi, train_obj_qnehvi, train_con_qnehvi, qnehvi_sampler
    )

    # delete after using
    del qnehvi_sampler

    # update training points
    train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
    train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
    train_con_qnehvi = torch.cat([train_con_qnehvi, new_con_qnehvi])
    
    # delete after using
    del new_x_qnehvi, new_obj_qnehvi, new_con_qnehvi

    # update progress
    # compute pareto front
    is_feas = (train_con_qnehvi <= 0).all(dim=-1)
    feas_train_obj = train_obj_qnehvi[is_feas]
    if feas_train_obj.shape[0] > 0:
        pareto_mask = is_non_dominated(feas_train_obj)
        pareto_y = feas_train_obj[pareto_mask]
        # compute hypervolume
        volume = hv.compute(torch.log(pareto_y))
    else:
        volume = 0.0
        
    hvs_qnehvi.append(volume)

    # reinitialize the models so they are ready for fitting on next iteration
    # Note: we find improved performance from not warm starting the model hyperparameters
    # using the hyperparameters from the previous iteration
    mll_qnehvi, model_qnehvi = initialize_model(
        train_x_qnehvi, train_obj_qnehvi, train_con_qnehvi
    )

    print(
        f"\nBatch {iteration:>2}: Hypervolume qNEHVI = "
        f"({hvs_qnehvi[-1]})"
    )


train_obj = pd.DataFrame(train_obj_qnehvi)
train_obj.to_csv('obj_config6.csv', index=False)
train_x = pd.DataFrame(train_x_qnehvi)
train_x.to_csv('x_config6.csv', index=False)
hvs = pd.DataFrame(hvs_qnehvi)
hvs.to_csv('hv_config6.csv', index=False)
