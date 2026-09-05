# Cell Performance Model Documentation

## Overview

The Cell Performance Model computes equilibrium KPIs (capacity, energy, N/P ratio, mass, energy density) from cell design inputs: electrode formulation (active materials, binders, conductive agents), geometry (mass loading, thickness, sheet count), and cell dimensions (Pouch or Cylindrical). It derives porosity, volume fractions, and mass breakdown from the formulation without requiring electrochemical simulation.

The model always runs a PyBaMM SPMe (Single Particle Model with electrolyte) calibration pipeline when electrode area > 0 (and PyBaMM is installed), producing nominal capacity, DCIR (supports charge/discharge via `dcir_direction`), and maximum 300 s power (supports charge/discharge via `power_direction`). The model supports extensive PyBaMM parameter overrides: electrolyte transport properties, active material diffusivity/particle size/kinetics/thermodynamics (MSMR), and electrode tortuosity. Custom OCV data can override PyBaMM defaults.

**Model Key**: `cell_performance`
**Model Type**: Internal
**Based on**: PyBaMM SPMe (Single Particle Model with electrolyte) lithium-ion model (Chen2020, Prada2013 for LFP)

## Model Theory

### Equilibrium Calculation (No PyBaMM)

The model computes:

1. **Coating density** (g/cm³): `ρ_coat = mass_loading_mg_cm² × 10 / thickness_um`
2. **Formulation density**: Volume additivity `1/ρ = Σ(f_i/ρ_i)` over active materials, binders, conductive agents
3. **Porosity**: `ε = 1 - (1/ρ_coat) × (mass_loading × AM_vol_frac_from_formulation)`
4. **Theoretical capacity**: From electrode area, mass loading, specific capacity, sheet count, N/P ratio
5. **Mass breakdown**: Jelly roll (foils + coatings + separator), electrolyte (pore fill), casing
6. **Capacity-Weighted Voltage**: For blended electrodes, voltage is weighted by the capacity contribution of each material (physically accurate integration of voltage profile).

### Formulation Validation

The model enforces two consistency checks per electrode:

1. **Mass fractions sum to 1**: `sum(AM mass_frac) + sum(binder mass_frac) + sum(conductive mass_frac) = 1` (validated at input via Pydantic; tolerance 1e-6).
2. **Volume fractions sum to 1**: `porosity + am_vol_frac + binder_vol_frac + cond_vol_frac = 1` (checked at runtime; sanity check for derived values).

Violations raise `ValueError` (mass) or `ValidationError` (mass, when using Pydantic validation).

### Blending Theory and OCV

For electrodes with multiple active materials (e.g., Graphite + SiOx anode, or NMC + LFP cathode), the model computes a composite OCV curve using **Volumetric Differential Integration**.

1. **Input**: Individual OCV curves ($V$ vs. $x_{sto}$) for each material.
2. **Differential Capacity**: $\frac{dX_{blend}}{dV} = \sum f_i \cdot \frac{C_{max,i}}{\rho_i} \cdot \frac{dx_i}{dV}$
   - $f_i$: Mass fraction of material $i$
   - $\rho_i$: Density of material $i$
   - $C_{max,i}$: Maximum theoretical concentration of material $i$
   - $dx_i/dV$: Derivative of stoichiometry with respect to voltage for material $i$
3. **Integration**: The composite stoichiometry $X_{blend}(V)$ is obtained by integrating the differential sum over the voltage window.

This ensures proper thermodynamic equilibrium between materials in parallel.

### MSMR Fitting (Multi-Structure Multi-Response)

When custom OCV data is provided, the model can optionally fit a Multi-Structure Multi-Response (MSMR) thermodynamic model to the data. This provides a physically-grounded analytical OCV function $U(x)$ suitable for PyBaMM.

$x(V) = \sum_j \frac{X_j}{1 + \exp\left(\frac{F(V - U^0_j)}{RT\kappa_j}\right)}$

- $U^0_j$: Equilibrium potential of site $j$
- $\kappa_j$: Shape factor (interaction parameter)
- $X_j$: Fraction of sites of type $j$

### Stoichiometry Convention

The model consistently uses **Lithiation Extent** for stoichiometry ($x$):
- **$x=1$**: Fully Lithiated
  - Cathode: Discharged state (Low Voltage, e.g., 3.0 V for NMC)
  - Anode: Charged state (Low Voltage, e.g., 0.01 V for Graphite)
- **$x=0$**: Fully Delithiated
  - Cathode: Charged state (High Voltage, e.g., 4.2 V for NMC)
  - Anode: Discharged state (High Voltage, e.g., 1.5 V for Graphite)

### PyBaMM Pipeline (always runs when electrode area > 0)

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Capacity calibration (0.05C)                               │
│     Charge → hold → discharge; scale electrode width to target   │
├─────────────────────────────────────────────────────────────────┤
│  2. C/3 discharge (1 s rest, CCCV charge + C/3 discharge)      │
│     → nominal capacity [Ah], nominal energy [Wh]                │
│     → c3_discharge_timeseries                                   │
├─────────────────────────────────────────────────────────────────┤
│  3. Rate capability tests (optional, per rate_capability_tests) │
│     Discharge: CCCV charge to upper_V → rest 10 s → CC/CP dch  │
│     Charge:    C/3 discharge to lower_V → rest 10 s → CC/CP ch │
│     → rate_capability_timeseries[label]                         │
├─────────────────────────────────────────────────────────────────┤
│  4. DCIR @ configurable SOC/temp, 10 s rest, 2C pulse 10 s     │
│     Supports charge/discharge via dcir_direction                 │
│     Runs only when perform_rpt=true                             │
│     → cell_dcir_10s_mohm                                        │
├─────────────────────────────────────────────────────────────────┤
│  5. Max power 300 s (bisection, lumped thermal)                 │
│     Supports charge/discharge via power_direction                │
│     Runs only when perform_rpt=true                             │
│     → cell_nominal_max_discharge_power_300s_W                   │
│     (anode potential + temperature safety limits)               │
└─────────────────────────────────────────────────────────────────┘
```

## Input Parameters

### Input Structure

The model uses a nested `CellPerformanceInput` schema:

- **cell_parameters** (`CellParametersInput`): Electrode design, formulation, dimensions, foils, separator
- **simulation_parameters** (`SimulationParameters`): Solver settings, mesh, RPT options (DCIR, power pulse)

### Required Electrode / Formulation (in `cell_parameters`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `positive_electrode_mass_loading_mg_cm2` | float | Positive electrode mass loading [mg/cm²] |
| `negative_electrode_mass_loading_mg_cm2` | float | Negative electrode mass loading [mg/cm²] |
| `positive_electrode_sheet_count` | float | Number of positive electrode sheets |
| `negative_electrode_sheet_count` | float | Number of negative electrode sheets |
| `positive_coating_thickness_um` | float | Positive coating thickness [µm] |
| `negative_coating_thickness_um` | float | Negative coating thickness [µm] |
| `positive_electrode_specific_heat_capacity_J_kg_K` | float | Positive electrode specific heat [J/kg/K] |
| `negative_electrode_specific_heat_capacity_J_kg_K` | float | Negative electrode specific heat [J/kg/K] |
| `positive_electrode_thermal_conductivity_W_m_K` | float | Positive electrode thermal conductivity [W/m/K] |
| `negative_electrode_thermal_conductivity_W_m_K` | float | Negative electrode thermal conductivity [W/m/K] |
| `positive_electrode_electronic_conductivity_S_m` | float | Positive electrode electronic conductivity [S/m] |
| `negative_electrode_electronic_conductivity_S_m` | float | Negative electrode electronic conductivity [S/m] |

### Formulation: Active Materials, Binders, Conductive Agents

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positive_electrode_active_materials` | list[ActiveMaterial] | NMC811 (96%) | Positive active material list |
| `negative_electrode_active_materials` | list[ActiveMaterial] | Graphite (98%) | Negative active material list |
| `positive_electrode_binders` | list[Binder] | PVDF (2%) | Positive binders |
| `negative_electrode_binders` | list[Binder] | CMC (1%), SBR (1%) | Negative binders |
| `positive_electrode_conductive_agents` | list[ConductiveAgent] | Carbon Black (2%) | Positive conductive agents |
| `negative_electrode_conductive_agents` | list[ConductiveAgent] | [] | Negative conductive agents |

**Formulation validation** (per electrode):
- **Mass fractions**: `sum(AM mass_frac) + sum(binder mass_frac) + sum(conductive mass_frac) = 1`
- **Volume fractions**: `porosity + am_vol_frac + binder_vol_frac + cond_vol_frac = 1` (checked at runtime)

### ActiveMaterial (per material)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | str | Yes | Material name (e.g. NMC811, Graphite, LFP) |
| `mass_fraction` | float | Yes | Mass fraction in electrode (0–1) |
| `density_g_cm3` | float | Yes | Density [g/cm³] |
| `specific_capacity_mAh_g` | float | Yes | Specific capacity [mAh/g] |
| `nominal_voltage_V` | float | Yes | Nominal voltage [V] (used for energy calc) |
| `soc_pct` | list[float] \| null | No | SOC [0–100%] for OCV (optional) |
| `ch_ocv_V` | list[float] \| null | No | OCV charge [V] |
| `dch_ocv_V` | list[float] \| null | No | OCV discharge [V] |
| `ocv_specific_capacity_mAh_g` | list[float] \| null | No | Specific capacity at each point [mAh/g] |
| `max_lithium_conc_mol_m3` | float \| null | No | Max Li concentration [mol/m³] for PyBaMM |
| `diffusivity_m2_s` | float \| IonicPropertyTable \| null | No | Particle diffusivity [m²/s]; scalar or tabulated |
| `particle_size_distribution` | ParticleSizeDistribution \| null | No | PSD; d50 → radius for PyBaMM |
| `exchange_current_density_A_m2` | float \| null | No | Exchange current density [A/m²] |
| `double_layer_capacitance_F_m2` | float \| null | No | Double-layer capacitance [F/m²] |

### ParticleSizeDistribution

| Field | Type | Description |
|-------|------|-------------|
| `d50_um` | float | Median diameter [µm]; used for PyBaMM radius = d50/2 |
| `d10_um` | float \| null | 10th percentile [µm] (optional) |
| `d90_um` | float \| null | 90th percentile [µm] (optional) |
| `dmin_um` | float \| null | Min diameter [µm] (optional) |
| `dmax_um` | float \| null | Max diameter [µm] (optional) |

### Binder / ConductiveAgent

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Component name |
| `mass_fraction` | float | Mass fraction (0–1) |
| `density_g_cm3` | float | Density [g/cm³] |

### Cell Dimensions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `form_factor` | str | "Pouch" | "Pouch", "Prismatic", "Cylindrical", "Coin" |
| `cell_width_mm` | float \| null | null | Cell width [mm] |
| `cell_height_mm` | float \| null | null | Cell height [mm] |
| `cell_thickness_mm` | float \| null | null | Cell thickness [mm] |
| `cell_diameter_mm` | float \| null | null | Cell diameter [mm] |

### Electrode Dimensions (Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positive_electrode_width_mm` | float \| null | null | Positive electrode width [mm] (optional, auto-derived if not provided) |
| `positive_electrode_height_mm` | float \| null | null | Positive electrode height [mm] (optional, auto-derived if not provided) |
| `negative_electrode_width_mm` | float \| null | null | Negative electrode width [mm] (optional, defaults to positive if not provided) |
| `negative_electrode_height_mm` | float \| null | null | Negative electrode height [mm] (optional, defaults to positive if not provided) |

If provided, these dimensions are used directly for electrode area calculation. If not provided, electrode area is calculated from cell dimensions using `volume_packing_ratio` and `electrode_overhang_mm`.

### Foils, Separator, Electrolyte

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positive_electrode_foil_thickness_um` | float | 14.0 | Positive foil [µm] |
| `negative_electrode_foil_thickness_um` | float | 6.0 | Negative foil [µm] |
| `positive_electrode_foil_density_g_cm3` | float | 2.7 | Al foil density |
| `negative_electrode_foil_density_g_cm3` | float | 8.96 | Cu foil density |
| `separator_thickness_um` | float | 12.0 | Separator [µm] |
| `separator_porosity` | float | 0.42 | Separator porosity |
| `separator_density_g_cm3` | float | 0.5 | Separator density [g/cm³] |
| `separator_sheet_count` | float \| null | null | Separator sheet count (optional, auto-derived if not provided) |
| `separator_sheet_width_mm` | float \| null | null | Separator sheet width [mm] (optional, auto-derived if not provided) |
| `separator_sheet_height_mm` | float \| null | null | Separator sheet height [mm] (optional, auto-derived if not provided) |
| `electrolyte_density_g_cm3` | float | 1.2 | Electrolyte density [g/cm³] (fallback when no `electrolyte`) |
| `electrolyte_fill_ratio` | float | 0.85 | Pore volume fill fraction |
| `electrolyte` | Electrolyte \| null | null | Optional electrolyte with composition + transport |
| `casing_thickness_mm` | float | 0.15 | Casing thickness [mm] |
| `casing_density_g_cm3` | float | 2.7 | Casing density [g/cm³] |
| `electrode_overhang_mm` | float | 1.0 | Electrode overhang [mm] |
| `jelly_roll_inner_diameter_mm` | float | 4.0 | Jelly roll inner diameter [mm] (cylindrical) |
| `volume_packing_ratio` | float | 0.95 | Volume packing ratio for JR in cell |

**Separator dimensions**: If not provided, automatically derived:
- `separator_sheet_count`: `positive_electrode_sheet_count + negative_electrode_sheet_count + 1`
- `separator_sheet_width_mm`: Uses `positive_electrode_width_mm` (or `negative_electrode_width_mm` if positive not available)
- `separator_sheet_height_mm`: Uses `positive_electrode_height_mm` (or `negative_electrode_height_mm` if positive not available)

### Electrolyte (optional)

If provided, properties override PyBaMM defaults.

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Electrolyte name |
| `composition` | ElectrolyteComposition | Solvents, salts, additives |
| `ionic_conductivity_S_m` | float \| IonicPropertyTable \| null | Ionic conductivity [S/m] |
| `ionic_diffusivity_m2_s` | float \| IonicPropertyTable \| null | Ionic diffusivity [m²/s] |
| `transference_number` | float \| IonicPropertyTable \| null | Cation transference number |
| `activity_coefficient` | float \| IonicPropertyTable \| null | Activity coefficient → Thermodynamic factor |

### Simulation Parameters (`SimulationParameters`)

Nested under `simulation_parameters`. RPT (reference performance test) settings use nested `rpt_dcir` and `rpt_power`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solver_atol` | float | 1e-3 | Solver absolute tolerance |
| `solver_rtol` | float | 1e-3 | Solver relative tolerance |
| `calibration_rate_C` | float | 0.05 | Capacity calibration C-rate |
| `mesh_resolution` | dict | x_n/x_s/x_p/r_n/r_p=10 | Mesh resolution (x=electrode/sep, r=particles) |
| `perform_rpt` | bool | false | Whether to run DCIR and max power tests |
| `rpt_dcir` | DCIRParameters | see below | DCIR pulse parameters |
| `rpt_power` | PowerPulseParameters | see below | Max power pulse parameters |
| `rate_capability_tests` | list[RateCapabilityTest] \| null | null | Optional list of CC or CP charge/discharge tests (see below); runs independently of `perform_rpt` |
| `cell_contact_resistance_Ohm` | float \| null | fallback | Cell contact resistance [Ω]; if null, falls back to `cell_parameters.cell_contact_resistance_Ohm`, and if that is also null, defaults to `1e-4` |
| `temperature_safety_threshold_K` | float | 363.15 | Max power temperature limit [K] |
| `anode_potential_safety_threshold_V` | float | 0.01 | Max power anode potential limit [V] |

**rpt_dcir** (DCIRParameters):
- `dcir_c_rate` (float): Pulse C-rate (default: 2.0)
- `dcir_direction` (str): "charge" or "discharge" (default: "discharge")
  - **Sign convention**: Discharge uses positive current, charge uses negative current (matches `prep_simulation_experiment`)
- `dcir_soc_pct` (float): SOC for pulse [0-100%] (default: 50.0)
- `dcir_temperature_K` (float): Temperature [K] (default: 298.15)
- `dcir_pulse_duration_s` (float): Pulse duration [s] (default: 10.0)
- `dcir_rest_s` (float): Rest time after pulse [s] (default: 1.0)

**rpt_power** (PowerPulseParameters):
- `power_level_W` (float): Initial power level for bisection search [W] (default: 2000.0)
- `power_duration_s` (float): Power pulse duration [s] (default: 300.0)
- `power_direction` (str): "charge" or "discharge" (default: "discharge")
  - **Sign convention**: Discharge uses positive power, charge uses negative power (matches `prep_simulation_experiment`)
  - Function returns signed power value (positive for discharge, negative for charge)
  - Output fields store absolute value for power densities
- `power_soc_pct` (float): SOC for pulse [0-100%] (default: 50.0)
- `power_temperature_K` (float): Temperature [K] (default: 298.15)

**rate_capability_tests** (list[RateCapabilityTest]):

Each entry configures one constant current (CC) or constant power (CP) charge or discharge test, run after the C/3 discharge and before DCIR/RPT. The test step timeseries is stored in `rate_capability_timeseries[label]`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `direction` | str | — | `"charge"` or `"discharge"` |
| `mode` | str | — | `"c_rate"` (constant current) or `"power"` (constant power) |
| `c_rate` | float \| null | null | C-rate for CC mode (required when `mode="c_rate"`) |
| `power_W` | float \| null | null | Power magnitude [W] for CP mode (required when `mode="power"`) |
| `temperature_K` | float | 298.15 | Ambient temperature [K] |
| `charge_mode` | str | `"cc"` | `"cc"` (constant-current only) or `"cccv"` (CC then CV hold). Only valid for `direction="charge"` and `mode="c_rate"` |
| `cv_cutoff_c_rate` | float | 0.05 | C-rate fraction at which the CV phase terminates (e.g. 0.05 → C/20). Only used when `charge_mode="cccv"`. Must be > 0 and ≤ 1 |
| `label` | str \| null | null | Key in `rate_capability_timeseries`; auto-generated as `{direction}_{mode}_{idx}` if omitted |

**Prep sequences:**
- **Discharge**: CCCV charge at C/3 to `upper_voltage_cutoff_V` → hold until C/50 → rest 10 s → CC/CP discharge to `lower_voltage_cutoff_V`
- **Charge**: C/3 CC discharge to `lower_voltage_cutoff_V` → rest 10 s → CC/CP charge to `upper_voltage_cutoff_V`

**Example:**
```json
"rate_capability_tests": [
  {"direction": "discharge", "mode": "c_rate", "c_rate": 0.333, "label": "C3_dch_25C"},
  {"direction": "discharge", "mode": "c_rate", "c_rate": 1.0,   "label": "1C_dch_25C"},
  {"direction": "discharge", "mode": "c_rate", "c_rate": 2.0,   "temperature_K": 318.15, "label": "2C_dch_45C"},
  {"direction": "charge",    "mode": "c_rate", "c_rate": 0.5,   "label": "C2_ch_25C"},
  {"direction": "discharge", "mode": "power",  "power_W": 500.0, "label": "500W_dch"}
]
```

### PyBaMM Configuration (in `simulation_parameters`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_pybamm_params` | str | "" | Base parameter set (e.g., 'Chen2020'). Empty string -> Use defaults/custom. |
| `upper_voltage_cutoff_V` | float \| null | 4.2 | Upper voltage [V] |
| `lower_voltage_cutoff_V` | float \| null | 3.0 | Lower voltage [V] |
| `cell_contact_resistance_Ohm` | float | 1e-4 | Contact resistance [Ω] |
| `cell_cooling_surface_area_m2` | float | 0.02 | Cooling surface area [m²] |
| `cell_heat_transfer_coefficient_W_m2_K` | float | 10.0 | Heat transfer coefficient [W/(m²·K)] |

### Electrode Tortuosity (in `cell_parameters`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positive_electrode_pore_tortuosity` | float \| null | null | Positive electrode tortuosity factor (electrolyte) |
| `negative_electrode_pore_tortuosity` | float \| null | null | Negative electrode tortuosity factor (electrolyte) |
| `separator_tortuosity` | float \| null | null | Separator tortuosity factor (electrolyte) |
| `positive_electrode_solid_tortuosity` | float \| null | null | Positive solid-phase tortuosity (defaults to pore if unset) |
| `negative_electrode_solid_tortuosity` | float \| null | null | Negative solid-phase tortuosity (defaults to pore if unset) |
| `cell_ocv` | CellOCVData \| null | null | Cell-level OCV (soc, specific_capacity_mAh_g, ch_ocv, dch_ocv) |

When any tortuosity is set, PyBaMM uses `transport efficiency: tortuosity factor` and these values override Bruggeman.

### CellOCVData (cell-level OCV)

| Field | Type | Description |
|-------|------|-------------|
| `soc_pct` | list[float] | SOC points [0–100%] |
| `specific_capacity_mAh_g` | list[float] | Capacity [mAh/g] at each SOC |
| `ch_ocv_V` | list[float] | Cell OCV charge [V] |
| `dch_ocv_V` | list[float] | Cell OCV discharge [V] |

Used for OCV100/OCV0 extraction when `use_pybamm_params` is empty.

## Output Structure (`CellPerformanceKPIs`)

The model returns `CellPerformanceKPIs` (extends `CellEquilibriumKPIs` with PyBaMM-derived fields).

### Equilibrium Outputs (always present)

| Field | Type | Description |
|-------|------|-------------|
| `cell_theoretical_capacity_Ah` | float | Equilibrium-derived capacity [Ah] |
| `cell_theoretical_energy_Wh` | float | Equilibrium-derived energy [Wh] |
| `cell_capacity_weighted_voltage_V` | float | Capacity-weighted average voltage [V] |
| `cell_n_p_ratio` | float | N/P capacity ratio |
| `cell_mass_g` | float | Total cell mass [g] |
| `cell_volume_L` | float | Cell volume [L] |
| `cell_theoretical_gravimetric_energy_density_Wh_kg` | float | [Wh/kg] |
| `cell_theoretical_volumetric_energy_density_Wh_L` | float | [Wh/L] |
| `stoichiometry_windows` | StoichiometryWindows | Electrode sto windows (sto_ca0/100, sto_an0/100) |
| `positive_electrode_porosity` | float | Positive electrode porosity |
| `negative_electrode_porosity` | float | Negative electrode porosity |
| `positive_electrode_volume_fraction` | float | Positive active material volume fraction |
| `negative_electrode_volume_fraction` | float | Negative active material volume fraction |
| `electrolyte_mass_g` | float | Electrolyte mass [g] |
| `electrolyte_mass_fraction` | float | Electrolyte mass fraction |
| `jelly_roll_mass_g` | float | Jelly roll mass [g] |
| `jelly_roll_mass_fraction` | float | Jelly roll mass fraction |
| `casing_mass_g` | float | Casing mass [g] |
| `casing_mass_fraction` | float | Casing mass fraction |
| `bill_of_materials` | BillOfMaterials | BOM: AM, binders, conductive, separator, foils, electrolyte, casing. Computed as part of equilibrium calculation (not PyBaMM). |

### PyBaMM Outputs (from calibration pipeline)

| Field | Type | Description |
|-------|------|-------------|
| `cell_nominal_capacity_Ah` | float \| null | PyBaMM nominal capacity from C/3 [Ah] |
| `cell_nominal_energy_Wh` | float \| null | PyBaMM nominal energy from C/3 [Wh] |
| `cell_nominal_gravimetric_energy_density_Wh_kg` | float \| null | [Wh/kg] |
| `cell_nominal_volumetric_energy_density_Wh_L` | float \| null | [Wh/L] |
| `cell_nominal_max_discharge_power_300s_W` | float \| null | Max 300 s power [W] (magnitude; direction controlled by `power_direction`) |
| `cell_nominal_gravimetric_power_density_W_kg` | float \| null | [W/kg] |
| `cell_nominal_volumetric_power_density_W_L` | float \| null | [W/L] |
| `cell_dcir_10s_mohm` | float \| null | DCIR at 10 s, 50% SOC, 25°C [mOhm] |
| `ocv100_V` | float \| null | Open-circuit voltage at 100% SOC [V] |
| `ocv0_V` | float \| null | Open-circuit voltage at 0% SOC [V] |
| `c3_discharge_timeseries` | TimeseriesOutput \| null | C/3 discharge timeseries (subsampled via RDP); excluded from copilot summary |
| `rate_capability_timeseries` | dict[str, TimeseriesOutput] \| null | Rate capability test timeseries keyed by label; excluded from copilot summary |

**TimeseriesOutput** (fields common to `c3_discharge_timeseries` and each entry in `rate_capability_timeseries`):

| Field | Type | Description |
|-------|------|-------------|
| `time_s` | list[float] | Time from start of test step [s] |
| `voltage_V` | list[float] | Terminal voltage [V] |
| `current_A` | list[float] | Current [A] (positive = discharge in PyBaMM) |
| `temperature_K` | list[float] | Spatially-averaged cell temperature [K] |
| `capacity_Ah` | list[float] | Cumulative capacity from step start [Ah] |
| `energy_Wh` | list[float] | Cumulative energy from step start [Wh] |
| `power_W` | list[float] | Instantaneous power [W] |
| `anode_potential_V` | list[float] | Negative electrode surface potential at separator [V] |
| `cc_capacity_Ah` | float \| null | CC-phase capacity [Ah]; populated for `charge_mode='cccv'` tests only |
| `total_capacity_Ah` | float \| null | Total CC+CV capacity [Ah]; populated for `charge_mode='cccv'` tests only |

### BillOfMaterials

| Field | Type | Description |
|-------|------|-------------|
| `active_materials` | list[BillOfMaterialsItem] | Active materials per electrode (name, electrode, mass_g, mass_fraction) |
| `binders` | list[BillOfMaterialsItem] | Binders per electrode |
| `conductive_agents` | list[BillOfMaterialsItem] | Conductive additives per electrode |
| `separator` | BillOfMaterialsItem \| null | Separator |
| `foils` | list[BillOfMaterialsItem] | Current collectors (Al, Cu) |
| `electrolyte` | BillOfMaterialsItem \| null | Electrolyte |
| `casing` | BillOfMaterialsItem \| null | Casing (universal enclosure) |

### BillOfMaterialsItem

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Material or component name |
| `electrode` | str \| null | "positive", "negative", or null |
| `mass_g` | float | Mass [g] |
| `mass_fraction` | float | Mass fraction of total cell |

## PyBaMM Parameter Overrides

When input fields are set, the model overrides PyBaMM parameters.
- **Electrolyte**: Conductivity, diffusivity, transference number, activity coefficient.
- **Active Material**: Diffusivity, particle size distribution (d50), exchange current density, double-layer capacitance.
- **Tortuosity**: Electrolyte and solid phase tortuosity.
- **OCV**: Custom OCV curves (MSMR or Interpolant) override default OCP.

## Model Configuration

### Parameter Sets
- **Chen2020**: Default for NMC/graphite.
- **Prada2013**: Auto-selected for LFP if `pybamm_parameter_set` is not specified.
- **Custom**: Use `use_pybamm_params=""` and provide input data.

### Solver
- **IDAKLUSolver**: Robust DAE solver.
- **Mesh**: Configurable resolution via `simulation_parameters`.

## References

- PyBaMM: https://pybamm.readthedocs.io/
- Chen2020 parameter set
- Prada2013 parameter set (LFP)
- MSMR: Multi-Structure Multi-Response Model for OCV fitting.

## Model Implementation

**File**: `backend/app/domains/models/cell_performance.py`
**Migration**: `backend/app/db/migrations/migration_037_seed_cell_performance_model.py`

## Recent Changes

### 2026-03-11 — Rate Capability Simulations

**New feature**: `rate_capability_tests` in `SimulationParameters` — an optional list of constant C-rate (CC) or constant power (CP) charge/discharge tests that run as pipeline step 3, after the C/3 discharge and before DCIR/RPT.

- **New type**: `RateCapabilityTest` (`direction`, `mode`, `c_rate`, `power_W`, `temperature_K`, `label`)
- **New output field**: `rate_capability_timeseries: dict[str, TimeseriesOutput] | null` in `CellPerformanceKPIs`
- **New private function**: `_run_rate_capability_pybamm` — builds a prep + test experiment, extracts only the test step timeseries, applies RDP subsampling
- `summarize_for_copilot` drops `rate_capability_timeseries` (alongside `c3_discharge_timeseries`) to keep copilot payload lean

### 2026-02-19

#### Bug Fixes
- **DCIR pulse sign convention**: Fixed to match `prep_simulation_experiment` - discharge uses positive current, charge uses negative current
- **Power pulse direction**: `power_direction` parameter now works correctly - discharge uses positive power, charge uses negative power
- **OCV extraction**: OCV values (`ocv100_V`, `ocv0_V`) are now correctly extracted even when capacity calibration fails

#### New Features
- **Electrode dimensions**: Added optional `positive_electrode_width_mm`, `positive_electrode_height_mm`, `negative_electrode_width_mm`, `negative_electrode_height_mm` fields for explicit electrode area specification
- **Separator dimensions**: Separator sheet dimensions (`separator_sheet_count`, `separator_sheet_width_mm`, `separator_sheet_height_mm`) are now optional and auto-derived if not provided
- **Power temperature**: Added `power_temperature_K` parameter to `PowerPulseParameters`

#### API Improvements
- DCIR and power pulse tests now support both charge and discharge directions via `dcir_direction` and `power_direction` parameters
- Sign conventions match `prep_simulation_experiment` throughout the codebase
