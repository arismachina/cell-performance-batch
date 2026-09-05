"""Cell Performance model — equilibrium KPIs + PyBaMM SPMe pipeline.

Blend-aware capacity-weighted OCV, MSMR fitting, explicit None/error
propagation, context-manager cleanup for PyBaMM sims.

VENDORED — do not hand-edit. This file is a verbatim copy of
``backend/app/domains/models/cell_performance.py`` from the Aris Protos
monorepo (protos-v2), taken at commit
``b5339610e0e44df648aa394e27165e68f5400ee2``. It is the single-design
model; the batch layer in :mod:`cell_performance_batch.batch` is the
only thing this repo adds. To pull a newer upstream revision, re-copy
the file and update ``UPSTREAM_COMMIT`` in ``__init__.py`` — keeping it
byte-identical is what makes batch results comparable to Protos'
built-in ``cell_performance`` model.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import logging
from math import pi, sqrt
from typing import Any, Literal, Protocol

import numpy as np
import pybamm
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.optimize import least_squares


class ProgressCallback(Protocol):
    """Protocol for progress callback functions.

    Expected signature: callback(progress_pct: int, message: str) -> None
    - progress_pct: Progress percentage (0-100)
    - message: Status message describing current operation
    """

    def __call__(self, progress_pct: int, message: str) -> None: ...


logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1 — SCHEMAS & I/O MODELS
# =============================================================================


class IonicPropertyTable(BaseModel):
    """Tabulated ionic transport property (conc, temp -> value)."""

    model_config = ConfigDict(extra="forbid")

    conc_mol_m3: list[float] = Field(..., description="Concentration [mol/m³]")
    temp_K: list[float] = Field(..., description="Temperature [K]")
    value: list[float] = Field(..., description="Property values (same length)")

    @model_validator(mode="after")
    def _validate_lengths(self) -> IonicPropertyTable:
        n = len(self.conc_mol_m3)
        if not (len(self.temp_K) == len(self.value) == n):
            raise ValueError("conc_mol_m3, temp_K, value must have equal length")
        return self


class ParticleSizeDistribution(BaseModel):
    """Particle size distribution. d50 used for PyBaMM radius."""

    model_config = ConfigDict(extra="forbid")

    dmin_um: float | None = Field(None, gt=0, description="Minimum diameter [µm]")
    d10_um: float | None = Field(None, gt=0, description="10th percentile [µm]")
    d50_um: float = Field(..., gt=0, description="Median diameter [µm]")
    d90_um: float | None = Field(None, gt=0, description="90th percentile [µm]")
    dmax_um: float | None = Field(None, gt=0, description="Maximum diameter [µm]")

    def radius_m(self) -> float:
        return (self.d50_um / 2.0) / 1e6


class ActiveMaterial(BaseModel):
    """Active material in electrode formulation. Optional OCV data for PyBaMM."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ..., description="Material name, e.g. 'NMC811', 'LFP', 'Graphite'"
    )
    mass_fraction: float = Field(
        ..., ge=0, le=1, description="Mass fraction in electrode formulation"
    )
    density_g_cm3: float = Field(..., gt=0, description="True density [g/cm³]")
    specific_capacity_mAh_g: float = Field(
        ..., ge=0, description="Reversible specific capacity [mAh/g]"
    )
    nominal_voltage_V: float = Field(..., description="Midpoint voltage vs Li/Li⁺ [V]")

    # OCV for PyBaMM override (use_pybamm_params=False)
    # Merged with CellOCVData column names
    soc_pct: list[float] | None = Field(
        None, min_length=2, description="State of charge [0-100%]"
    )
    ch_ocv_V: list[float] | None = Field(None, description="OCV during charge [V]")
    dch_ocv_V: list[float] | None = Field(None, description="OCV during discharge [V]")
    ocv_specific_capacity_mAh_g: list[float] | None = Field(
        None, description="Specific capacity [mAh/g] per point"
    )
    max_lithium_conc_mol_m3: float | None = Field(
        None, gt=0, description="Max Li concentration [mol/m³]"
    )

    # Particle / kinetics overrides
    diffusivity_m2_s: float | IonicPropertyTable | None = Field(
        None, description="Particle diffusivity [m²/s]"
    )
    particle_size_distribution: ParticleSizeDistribution | None = Field(
        None, description="PSD (d50 → radius)"
    )
    exchange_current_density_A_m2: float | None = Field(
        None, gt=0, description="Electrode exchange current density [A/m²]"
    )
    double_layer_capacitance_F_m2: float | None = Field(
        None, gt=0, description="Double-layer capacitance [F/m²]"
    )

    @model_validator(mode="after")
    def _validate_ocv_consistency(self) -> ActiveMaterial:
        ocv_fields = [self.soc_pct, self.ch_ocv_V, self.dch_ocv_V]
        if any(f is not None for f in ocv_fields):
            if any(f is None for f in ocv_fields):
                raise ValueError(
                    "soc_pct, ch_ocv_V, dch_ocv_V must all be set or all None"
                )
            if not (len(self.soc_pct) == len(self.ch_ocv_V) == len(self.dch_ocv_V)):
                raise ValueError("soc_pct, ch_ocv_V, dch_ocv_V must have equal length")
        return self

    def has_ocv_data(self) -> bool:
        return (
            self.soc_pct is not None
            and self.ch_ocv_V is not None
            and self.dch_ocv_V is not None
            and len(self.soc_pct) >= 2
        )


class Binder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Binder name")
    mass_fraction: float = Field(
        ..., ge=0, le=1, description="Mass fraction in electrode formulation"
    )
    density_g_cm3: float = Field(..., gt=0, description="Density [g/cm³]")


class ConductiveAgent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Conductive agent name")
    mass_fraction: float = Field(
        ..., ge=0, le=1, description="Mass fraction in electrode formulation"
    )
    density_g_cm3: float = Field(..., gt=0, description="Density [g/cm³]")


class Solvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Solvent name")
    density_g_cm3: float = Field(..., gt=0, description="Density [g/cm³]")
    molar_vol_cm3_mol: float = Field(..., gt=0, description="Molar volume [cm³/mol]")
    molar_mass_g_mol: float = Field(..., gt=0, description="Molar mass [g/mol]")
    vol_frac: float = Field(
        ..., ge=0, le=1, description="Volume fraction in electrolyte formulation"
    )


class Salt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Salt name")
    density_g_cm3: float = Field(..., gt=0, description="Density [g/cm³]")
    molar_vol_cm3_mol: float = Field(..., gt=0, description="Molar volume [cm³/mol]")
    molar_mass_g_mol: float = Field(..., gt=0, description="Molar mass [g/mol]")
    mass_fraction: float = Field(
        ..., ge=0, le=1, description="Mass fraction in electrolyte formulation"
    )


class Additive(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Additive name")
    density_g_cm3: float = Field(..., gt=0)
    molar_vol_cm3_mol: float = Field(..., gt=0)
    molar_mass_g_mol: float = Field(..., gt=0)
    mass_fraction: float = Field(..., ge=0, le=1)


class ElectrolyteComposition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    solvents: list[Solvent] = Field(default_factory=list)
    salts: list[Salt] = Field(default_factory=list)
    additives: list[Additive] = Field(default_factory=list)


class Electrolyte(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Electrolyte name")
    composition: ElectrolyteComposition = Field(...)
    ionic_conductivity_S_m: float | IonicPropertyTable | None = Field(None)
    ionic_diffusivity_m2_s: float | IonicPropertyTable | None = Field(None)
    transference_number: float | IonicPropertyTable | None = Field(None)
    activity_coefficient: float | IonicPropertyTable | None = Field(None)


# --- Simulation & experiment configuration ---
class DCIRParameters(BaseModel):
    dcir_c_rate: float = Field(..., gt=0, description="DCIR pulse C-rate")
    dcir_direction: Literal["charge", "discharge"] = Field(
        ..., description="DCIR pulse direction"
    )
    dcir_soc_pct: float = Field(..., ge=0, le=100, description="DCIR pulse SOC [%]")
    dcir_temperature_K: float = Field(
        ..., gt=0, description="DCIR pulse temperature [K]"
    )
    dcir_pulse_duration_s: float = Field(
        ..., gt=0, description="DCIR pulse duration [s]"
    )
    dcir_rest_s: float = Field(
        default=1.0,
        gt=0,
        description="Rest time before DCIR pulse (pre-pulse rest) [s]",
    )


class PowerPulseParameters(BaseModel):
    power_level_W: float = Field(..., gt=0, description="Power pulse level [W]")
    power_duration_s: float = Field(..., gt=0, description="Power pulse duration [s]")
    power_direction: Literal["charge", "discharge"] = Field(
        ..., description="Power pulse direction"
    )
    power_soc_pct: float = Field(..., ge=0, le=100, description="Power pulse SOC [%]")
    power_temperature_K: float = Field(
        ..., gt=0, description="Power pulse temperature [K]"
    )


class ExperimentTermination(BaseModel):
    """Custom termination condition applied to every step in a PyBaMM experiment.

    At each solver step, PyBaMM evaluates an event function that is configured
    based on ``comparison``:

    - ``comparison='less_than'``: ``f(variables) = variables[variable] - threshold``
    - ``comparison='greater_than'``: ``f(variables) = threshold - variables[variable]``

    In both cases, the step terminates when the event function becomes
    non-positive (``f(variables) <= 0``), i.e. when the monitored variable
    has crossed the specified threshold in the requested direction.
    Example — stop when anode potential drops below 0 V:
        ExperimentTermination(variable="Anode potential [V]", threshold=0.0)
    """

    model_config = ConfigDict(extra="forbid")

    variable: str = Field(
        ...,
        description=(
            "PyBaMM variable name to monitor, "
            "e.g. 'Anode potential [V]' or 'Cell temperature [K]'."
        ),
    )
    threshold: float = Field(..., description="Threshold value for the termination.")
    comparison: Literal["less_than", "greater_than"] = Field(
        "less_than",
        description=(
            "'less_than': terminate when variable < threshold. "
            "'greater_than': terminate when variable > threshold."
        ),
    )
    name: str | None = Field(
        None,
        description="Human-readable name used in logging; auto-generated if None.",
    )


class ExperimentConfig(BaseModel):
    """Configuration for a multi-step / multi-cycle PyBaMM experiment.

    ``steps`` defines the sequence of operations in a single cycle, expressed
    as PyBaMM step strings (e.g. ``"Discharge at 1C until 3.0 V"``).  The
    cycle is repeated ``n_cycles`` times.

    Example (3-cycle charge-discharge):
        ExperimentConfig(
            label="cycle_aging",
            steps=[
                "Discharge at 1C until 3.0 V",
                "Rest for 10 minutes",
                "Charge at 1C until 4.2 V",
                "Hold at 4.2 V until C/20",
                "Rest for 10 minutes",
            ],
            n_cycles=3,
        )
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., description="Key used in the output dict.")
    steps: list[str | DriveStepConfig | EISStepConfig] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of steps forming one cycle.  Each element is either a "
            "PyBaMM step string, a DriveStepConfig (time-varying profile), or an "
            "EISStepConfig (impedance snapshot)."
        ),
    )
    n_cycles: int = Field(..., ge=1, description="Number of times to repeat the cycle.")
    terminations: list[ExperimentTermination] | None = Field(
        default=None,
        description=(
            "Optional custom termination conditions applied to every step. "
            "A triggered condition ends the current step only (PyBaMM step-level "
            "semantics); the experiment continues with subsequent steps and cycles."
        ),
    )
    period_s: float = Field(..., gt=0, description="Solver output sampling period [s].")
    temperature_K: float = Field(
        ..., gt=0, description="Ambient temperature for this experiment [K]."
    )
    initial_soc_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Initial state of charge [%] before the experiment begins.",
    )


class ExperimentCycleSummary(BaseModel):
    """Per-cycle summary extracted from a completed experiment."""

    model_config = ConfigDict(extra="forbid")

    cycle_number: int = Field(..., description="1-indexed cycle number.")
    discharge_capacity_Ah: float | None = Field(
        None, description="Total discharge throughput in this cycle [Ah]."
    )
    charge_capacity_Ah: float | None = Field(
        None, description="Total charge throughput in this cycle [Ah]."
    )
    coulombic_efficiency_pct: float | None = Field(
        None,
        description=(
            "Coulombic efficiency for this cycle [%]: "
            "100 * discharge_capacity / charge_capacity."
        ),
    )
    total_capacity_throughput_Ah: float | None = Field(
        None,
        description=(
            "Total capacity throughput (sum of |Ah| across all steps) in this cycle [Ah]."
        ),
    )
    total_energy_throughput_Wh: float | None = Field(
        None,
        description=(
            "Total energy throughput (sum of |Wh| across all steps) in this cycle [Wh]."
        ),
    )
    min_voltage_V: float | None = Field(
        None, description="Minimum terminal voltage in this cycle [V]."
    )
    max_voltage_V: float | None = Field(
        None, description="Maximum terminal voltage in this cycle [V]."
    )
    max_temperature_K: float | None = Field(
        None, description="Maximum cell temperature in this cycle [K]."
    )


class ExperimentOutput(BaseModel):
    """Full output of a completed PyBaMM experiment."""

    model_config = ConfigDict(extra="forbid")

    timeseries: TimeseriesOutput = Field(
        ...,
        description=(
            "Concatenated timeseries from all cycles and steps (subsampled via RDP)."
        ),
    )
    cycle_summaries: list[ExperimentCycleSummary] = Field(
        default_factory=list,
        description="Per-cycle capacity and efficiency summaries.",
    )
    n_cycles_completed: int = Field(
        ..., description="Number of full cycles that completed before any termination."
    )
    total_capacity_throughput_Ah: float | None = Field(
        None,
        description="Total capacity throughput across all cycles and steps [Ah].",
    )
    total_energy_throughput_Wh: float | None = Field(
        None,
        description="Total energy throughput across all cycles and steps [Wh].",
    )
    eis_measurements: list[EISOutput] = Field(
        default_factory=list,
        description=(
            "EIS snapshots taken at each EIS step across all cycles, "
            "in execution order."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal warnings raised during experiment execution "
            "(e.g. missing optional dependencies that caused steps to be skipped)."
        ),
    )


class DriveStepConfig(BaseModel):
    """Time-varying drive cycle step using a (time, value) profile.

    Exactly one of ``current_A``, ``power_W``, or ``c_rate`` must be supplied.
    The profile length must match ``time_s``.

    Example — 30-second power pulse followed by zero:
        DriveStepConfig(
            kind="drive_cycle",
            time_s=[0, 30, 30.01, 60],
            power_W=[50.0, 50.0, 0.0, 0.0],
        )
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["drive_cycle"] = "drive_cycle"
    time_s: list[float] = Field(..., min_length=2, description="Time points [s].")
    current_A: list[float] | None = Field(
        None, description="Current profile [A]. Positive = discharge."
    )
    power_W: list[float] | None = Field(
        None, description="Power profile [W]. Positive = discharge."
    )
    c_rate: list[float] | None = Field(
        None, description="C-rate profile. Positive = discharge."
    )

    @model_validator(mode="after")
    def _validate_profile(self) -> DriveStepConfig:
        import math

        profiles = [
            p for p in (self.current_A, self.power_W, self.c_rate) if p is not None
        ]
        if len(profiles) == 0:
            raise ValueError(
                "Exactly one of current_A, power_W, or c_rate must be provided."
            )
        if len(profiles) > 1:
            raise ValueError(
                "Only one of current_A, power_W, or c_rate may be provided."
            )
        n = len(self.time_s)
        if len(profiles[0]) != n:
            raise ValueError("Profile length must match time_s length.")
        t = self.time_s
        if not all(math.isfinite(v) and v >= 0 for v in t):
            raise ValueError("time_s values must be finite and non-negative.")
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                raise ValueError(
                    f"time_s must be strictly increasing; "
                    f"got {t[i - 1]} then {t[i]} at index {i}."
                )
        profile = profiles[0]
        if not all(math.isfinite(v) for v in profile):
            raise ValueError("Profile values must all be finite (no NaN or ±inf).")
        return self


class EISStepConfig(BaseModel):
    """EIS snapshot step — runs a frequency-domain impedance sweep at the current
    battery state without advancing simulation time.

    Requires ``pybammeis`` to be installed (``pip install pybammeis``).
    If the package is absent the step is skipped and a warning is logged /
    surfaced in ``ExperimentOutput.warnings``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["eis"] = "eis"
    freq_min_hz: float = Field(1e-4, gt=0, description="Minimum frequency [Hz].")
    freq_max_hz: float = Field(1e4, gt=0, description="Maximum frequency [Hz].")
    n_frequencies: int = Field(
        30, ge=2, description="Number of log-spaced frequency points."
    )
    method: Literal["direct", "prebicgstab", "bicgstab"] = Field(
        "direct", description="Linear solver method used by pybammeis."
    )

    @model_validator(mode="after")
    def _validate_freq_range(self) -> EISStepConfig:
        if self.freq_min_hz >= self.freq_max_hz:
            raise ValueError("freq_min_hz must be less than freq_max_hz.")
        return self


class EISOutput(BaseModel):
    """Result of a single EIS measurement."""

    model_config = ConfigDict(extra="forbid")

    frequencies_hz: list[float] = Field(..., description="Frequency points [Hz].")
    z_real_ohm: list[float] = Field(..., description="Real part of impedance [Ω].")
    z_imag_ohm: list[float] = Field(
        ...,
        description=(
            "Imaginary part of impedance [Ω]. "
            "Negative values indicate capacitive behaviour."
        ),
    )
    z_magnitude_ohm: list[float] = Field(..., description="|Z| magnitude [Ω].")
    z_phase_deg: list[float] = Field(..., description="Phase angle [°].")
    cycle_number: int = Field(
        ..., description="1-indexed cycle in which this was taken."
    )
    step_index: int = Field(
        ...,
        description="0-indexed position of this EIS step within the full step list.",
    )
    soc_pct: float | None = Field(
        None, description="Estimated SOC at measurement time [%]."
    )


class StoichiometryWindows(BaseModel):
    """Electrode stoichiometry windows from equilibrium + coulombic loss."""

    model_config = ConfigDict(extra="forbid")

    sto_ca0: float = Field(..., ge=0, le=1, description="Cathode sto at 0% SOC")
    sto_ca100: float = Field(..., ge=0, le=1, description="Cathode sto at 100% SOC")
    sto_an0: float = Field(..., ge=0, le=1, description="Anode sto at 0% SOC")
    sto_an100: float = Field(..., ge=0, le=1, description="Anode sto at 100% SOC")


class CellOCVData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    soc_pct: list[float] = Field(
        ..., min_length=2, description="State of charge [0-100%]"
    )
    specific_capacity_mAh_g: list[float] = Field(
        ..., description="Specific capacity [mAh/g]"
    )
    ch_ocv_V: list[float] = Field(..., description="Charge OCV [V]")
    dch_ocv_V: list[float] = Field(..., description="Discharge OCV [V]")

    @model_validator(mode="after")
    def _validate_lengths(self) -> CellOCVData:
        n = len(self.soc_pct)
        if not (
            len(self.specific_capacity_mAh_g)
            == len(self.ch_ocv_V)
            == len(self.dch_ocv_V)
            == n
        ):
            raise ValueError(
                "soc_pct, specific_capacity_mAh_g, ch_ocv_V, dch_ocv_V must have equal length"
            )
        return self


# --- Bill of materials ---


class BillOfMaterialsItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Material name")
    electrode: str | None = Field(..., description="Electrode type")
    mass_g: float = Field(..., ge=0, description="Mass [g]")
    mass_fraction: float = Field(..., ge=0, le=1, description="Mass fraction [0-1]")


class BillOfMaterials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active_materials: list[BillOfMaterialsItem] = Field(
        default_factory=list, description="Active materials in both electrodes"
    )
    binders: list[BillOfMaterialsItem] = Field(
        default_factory=list, description="Binders in both electrodes"
    )
    conductive_agents: list[BillOfMaterialsItem] = Field(
        default_factory=list, description="Conductive agents in both electrodes"
    )
    separator: BillOfMaterialsItem | None = Field(
        default=None, description="Separator material"
    )
    foils: list[BillOfMaterialsItem] = Field(
        default_factory=list, description="Current collector foils"
    )
    electrolyte: BillOfMaterialsItem | None = Field(
        default=None, description="Electrolyte material"
    )
    casing: BillOfMaterialsItem | None = Field(
        default=None, description="Casing material"
    )


# --- Cell parameters (main input model) ---
class CellParametersInput(BaseModel):
    """Full structured input - used internally and by typed callers."""

    model_config = ConfigDict(extra="forbid")

    # Electrode / formulation
    positive_electrode_mass_loading_mg_cm2: float = Field(
        ..., description="Positive mass loading [mg/cm²]"
    )
    negative_electrode_mass_loading_mg_cm2: float = Field(
        ..., description="Negative mass loading [mg/cm²]"
    )
    positive_electrode_sheet_count: int = Field(..., description="Positive sheet count")
    negative_electrode_sheet_count: int = Field(..., description="Negative sheet count")
    jelly_roll_count: int = Field(..., description="Jelly roll count")
    electrode_coating_side_count: int = Field(
        ..., description="Number of coating sides for the electrode"
    )

    # Dimensions
    form_factor: Literal["Pouch", "Prismatic", "Cylindrical", "Coin"] = Field(
        ..., description="Pouch, Prismatic, Coin or Cylindrical"
    )
    cell_width_mm: float | None = Field(
        None, description="Cell width (or diameter for cylindrical/coin) [mm]"
    )
    cell_height_mm: float | None = Field(None, description="Cell height [mm]")
    cell_thickness_mm: float | None = Field(None, description="Cell thickness [mm]")
    cell_diameter_mm: float | None = Field(None, description="Cell diameter [mm]")

    # Electrode dimensions (if provided, used directly; otherwise calculated from cell dimensions)
    positive_electrode_width_mm: float | None = Field(
        None,
        description="Positive electrode width [mm]. If provided, used directly for area calculation instead of deriving from cell dimensions. Also used for negative electrode if negative-specific dimensions not provided.",
    )
    positive_electrode_height_mm: float | None = Field(
        None,
        description="Positive electrode height [mm]. If provided, used directly for area calculation instead of deriving from cell dimensions. Also used for negative electrode if negative-specific dimensions not provided.",
    )
    negative_electrode_width_mm: float | None = Field(
        None,
        description="Negative electrode width [mm]. If provided, used directly for area calculation instead of deriving from cell dimensions. If not provided, positive electrode width is used.",
    )
    negative_electrode_height_mm: float | None = Field(
        None,
        description="Negative electrode height [mm]. If provided, used directly for area calculation instead of deriving from cell dimensions. If not provided, positive electrode height is used.",
    )

    # Electrode
    positive_electrode_specific_heat_capacity_J_kg_K: float = Field(
        ..., description="Positive electrode specific heat capacity [J/kg/K]"
    )
    negative_electrode_specific_heat_capacity_J_kg_K: float = Field(
        ..., description="Negative electrode specific heat capacity [J/kg/K]"
    )
    positive_electrode_thermal_conductivity_W_m_K: float = Field(
        ..., description="Positive electrode thermal conductivity [W/m/K]"
    )
    negative_electrode_thermal_conductivity_W_m_K: float = Field(
        ..., description="Negative electrode thermal conductivity [W/m/K]"
    )
    positive_electrode_electronic_conductivity_S_m: float = Field(
        ..., description="Positive electrode electronic conductivity [S/m]"
    )
    negative_electrode_electronic_conductivity_S_m: float = Field(
        ..., description="Negative electrode electronic conductivity [S/m]"
    )
    # Coating
    positive_coating_thickness_um: float = Field(
        ..., description="Positive electrode coating thickness [µm]"
    )
    negative_coating_thickness_um: float = Field(
        ..., description="Negative electrode coating thickness [µm]"
    )

    # Formulation
    positive_electrode_active_materials: list[ActiveMaterial] = Field(
        ...,
        description="List of active materials in positive electrode formulation",
    )
    negative_electrode_active_materials: list[ActiveMaterial] = Field(
        ...,
        description="List of active materials in negative electrode formulation",
    )
    positive_electrode_binders: list[Binder] = Field(
        ...,
        description="List of binders in positive electrode formulation",
    )
    negative_electrode_binders: list[Binder] = Field(
        ...,
        description="List of binders in negative electrode formulation",
    )
    positive_electrode_conductive_agents: list[ConductiveAgent] = Field(
        ...,
        description="List of conductive agents in positive electrode formulation",
    )
    negative_electrode_conductive_agents: list[ConductiveAgent] = Field(
        ...,
        description="List of conductive agents in negative electrode formulation (can be empty)",
    )

    # Foil
    positive_electrode_foil_thickness_um: float = Field(
        ..., description="Positive electrode foil thickness [µm]"
    )
    negative_electrode_foil_thickness_um: float = Field(
        ..., description="Negative electrode foil thickness [µm]"
    )
    positive_electrode_foil_density_g_cm3: float = Field(
        ..., description="Positive electrode foil density [g/cm³]"
    )
    negative_electrode_foil_density_g_cm3: float = Field(
        ..., description="Negative electrode foil density [g/cm³]"
    )
    positive_electrode_foil_electronic_conductivity_S_m: float = Field(
        ...,
        description="Positive electrode foil electronic conductivity [S/m]",
    )
    negative_electrode_foil_electronic_conductivity_S_m: float = Field(
        ...,
        description="Negative electrode foil electronic conductivity [S/m]",
    )
    positive_electrode_foil_specific_heat_capacity_J_kg_K: float = Field(
        ...,
        description="Positive electrode foil specific heat capacity [J/kg/K]",
    )
    negative_electrode_foil_specific_heat_capacity_J_kg_K: float = Field(
        ...,
        description="Negative electrode foil specific heat capacity [J/kg/K]",
    )
    positive_electrode_foil_thermal_conductivity_W_m_K: float = Field(
        ...,
        description="Positive electrode foil thermal conductivity [W/m/K]",
    )
    negative_electrode_foil_thermal_conductivity_W_m_K: float = Field(
        ...,
        description="Negative electrode foil thermal conductivity [W/m/K]",
    )

    # Separator
    separator_sheet_count: float = Field(
        ...,
        description="Separator sheet count",
    )
    separator_sheet_width_mm: float = Field(
        ...,
        description="Separator sheet width [mm]",
    )
    separator_sheet_height_mm: float = Field(
        ...,
        description="Separator sheet height [mm]",
    )
    separator_thickness_um: float = Field(..., description="Separator thickness [µm]")
    separator_porosity: float = Field(..., description="Separator porosity")
    separator_density_g_cm3: float = Field(..., description="Separator density [g/cm³]")
    separator_specific_heat_capacity_J_kg_K: float = Field(
        ..., description="Separator specific heat capacity [J/kg/K]"
    )
    separator_thermal_conductivity_W_m_K: float = Field(
        ..., description="Separator thermal conductivity [W/m/K]"
    )
    electrolyte_density_g_cm3: float = Field(
        ..., description="Electrolyte density [g/cm³]"
    )
    electrolyte_fill_ratio: float = Field(..., description="Electrolyte fill ratio")
    electrolyte: Electrolyte | None = None
    casing_thickness_mm: float = Field(..., description="Casing thickness [mm]")
    casing_density_g_cm3: float = Field(..., description="Casing density [g/cm³]")
    electrode_overhang_mm: float = Field(
        ..., ge=0, description="Electrode overhang [mm]"
    )
    jelly_roll_inner_diameter_mm: float = Field(
        ..., ge=0, description="Jelly roll inner diameter [mm]"
    )
    volume_packing_ratio: float = Field(
        ..., ge=0, le=1, description="Volume packing ratio"
    )

    # Cell OCV data and simulation configuration
    cell_ocv: CellOCVData | None = None
    positive_electrode_pore_tortuosity: float | None = Field(
        None,
        gt=0,
        description="Tortuosity factor for positive electrode (applied to both electrolyte and solid phases unless solid-phase override is set)",
    )
    negative_electrode_pore_tortuosity: float | None = Field(
        None,
        gt=0,
        description="Tortuosity factor for negative electrode (applied to both electrolyte and solid phases unless solid-phase override is set)",
    )
    separator_tortuosity: float | None = Field(None, gt=0)
    # Optional: separate solid-phase tortuosity (defaults to same as electrolyte-phase)
    positive_electrode_solid_tortuosity: float | None = Field(
        None,
        gt=0,
        description="Solid-phase tortuosity for positive electrode (if different from electrolyte-phase)",
    )
    negative_electrode_solid_tortuosity: float | None = Field(
        None,
        gt=0,
        description="Solid-phase tortuosity for negative electrode (if different from electrolyte-phase)",
    )
    upper_voltage_cutoff_V: float = Field(
        ..., gt=0, description="Upper voltage cutoff for simulations [V]"
    )
    lower_voltage_cutoff_V: float = Field(
        ..., ge=0, description="Lower voltage cutoff for simulations [V]"
    )
    cell_contact_resistance_Ohm: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Cell contact resistance [Ohm]. Used as fallback when "
            "simulation_parameters.cell_contact_resistance_Ohm is not set."
        ),
    )

    @model_validator(mode="after")
    def _validate_formulation_mass_fractions_sum_to_one(self) -> CellParametersInput:
        tol = 1e-6

        def _check(electrode: str, am: list, binders: list, cond: list) -> None:
            total = (
                sum(m.mass_fraction for m in am)
                + sum(b.mass_fraction for b in binders)
                + sum(c.mass_fraction for c in cond)
            )
            if abs(total - 1.0) >= tol:
                raise ValueError(
                    f"{electrode} electrode: mass fractions must sum to 1, got {total:.10f}"
                )

        _check(
            "Positive",
            self.positive_electrode_active_materials,
            self.positive_electrode_binders,
            self.positive_electrode_conductive_agents,
        )
        _check(
            "Negative",
            self.negative_electrode_active_materials,
            self.negative_electrode_binders,
            self.negative_electrode_conductive_agents,
        )
        return self


class SimulationParameters(BaseModel):
    """Simulation configuration for PyBaMM pipeline.

    Note: This class mixes multiple concerns:
    - Simulation configuration (solver tolerances, mesh resolution)
    - RPT-specific parameters (rpt_dcir, rpt_power)
    - OCV model selection (positive_electrode_ocv_model, negative_electrode_ocv_model)
    - Safety thresholds (anode_potential_safety_threshold_V, temperature_safety_threshold_K)
    - Voltage cutoffs (upper_voltage_cutoff_V, lower_voltage_cutoff_V)

    Voltage cutoff duplication: upper_voltage_cutoff_V and lower_voltage_cutoff_V also exist
    in CellParametersInput. The values from SimulationParameters are used in PyBaMM simulations
    (calibration, C/3 discharge, DCIR, power tests), while CellParametersInput values are
    used in equilibrium calculations. Ensure consistency between the two if both are provided.
    """

    model_config = ConfigDict(populate_by_name=True, validate_default=True)

    solver_atol: float = Field(1e-3, gt=0, description="Solver absolute tolerance")
    solver_rtol: float = Field(1e-3, gt=0, description="Solver relative tolerance")
    calibration_rate_C: float = Field(
        ..., gt=0, description="Model capacity calibration rate [C]"
    )
    mesh_resolution: dict[str, int] = Field(
        default_factory=lambda: {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10},
    )
    data_sampling_period_s: float = Field(
        0.1, gt=0, description="Data sampling period [s]"
    )
    first_cycle_coulombic_loss_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="First-cycle coulombic loss for initial electrode balancing[%]",
    )
    ambient_temperature_K: float = Field(
        ..., gt=0, description="Ambient temperature for simulation [K]"
    )
    initial_cell_temperature_K: float | None = Field(
        None, gt=0, description="Initial cell temperature for simulation [K]"
    )
    reference_cell_temperature_K: float = Field(
        ..., gt=0, description="Reference cell temperature for performance test [K]"
    )
    perform_rpt: bool = Field(
        default=False, description="Whether to perform reference performance test (RPT)"
    )
    rpt_dcir: DCIRParameters | None = Field(
        ...,
        description="Parameters for DCIR measurement pulse for reference performance test (RPT)",
    )
    rpt_power: PowerPulseParameters | None = Field(
        ...,
        description="Parameters for power measurement pulse for reference performance test (RPT)",
    )
    start_soc_pct: float | None = Field(
        ...,
        ge=0,
        le=100,
        description="Starting state of charge for load cycle simulation [%]",
    )

    # PyBaMM calibration and simulation settings
    use_pybamm_params: str | None = Field(
        default="",
        description="PyBaMM parameter set to use (e.g. 'Chen2020') or 'custom' string to use custom params",
    )
    upper_voltage_cutoff_V: float = Field(
        ..., gt=0, description="Upper voltage cutoff for simulations [V]"
    )
    lower_voltage_cutoff_V: float = Field(
        ..., ge=0, description="Lower voltage cutoff for simulations [V]"
    )
    cell_contact_resistance_Ohm: float | None = Field(
        ...,
        ge=0,
        description="Cell contact resistance [Ohm]. If None, falls back to cell_parameters value (default 1e-4).",
    )
    cell_cooling_surface_area_m2: float = Field(
        ..., gt=0, description="Cell cooling surface area [m²]"
    )
    cell_heat_transfer_coefficient_W_m2_K: float = Field(
        ..., gt=0, description="Cell heat transfer coefficient [W/m²/K]"
    )
    anode_potential_safety_threshold_V: float = Field(
        ..., description="Anode potential safety threshold [V]"
    )
    temperature_safety_threshold_K: float = Field(
        ..., gt=0, description="Temperature safety threshold [K]"
    )

    positive_electrode_ocv_model: (
        Literal["msmr", "polynomial", "interpolant"] | None
    ) = Field("polynomial", description="OCV model for positive electrode in PyBaMM")
    negative_electrode_ocv_model: (
        Literal["msmr", "polynomial", "interpolant"] | None
    ) = Field("polynomial", description="OCV model for negative electrode in PyBaMM")
    particle_diffusion_model: Literal[
        "Fickian diffusion", "uniform profile", "quadratic profile", "quartic profile"
    ] = Field(
        "Fickian diffusion",
        description=(
            "Particle diffusion model for PyBaMM simulations. "
            "'Fickian diffusion' (default) solves the full diffusion PDE inside each "
            "particle. 'uniform profile' assumes spatially uniform concentration "
            "(fastest, lowest accuracy). 'quadratic profile' and 'quartic profile' are "
            "polynomial approximations that trade some accuracy for speed."
        ),
    )
    negative_electrode_phases: Literal[1, 2] = Field(
        1,
        description=(
            "Number of particle phases in the negative electrode. "
            "1 = single-phase (default). 2 = composite (e.g. graphite-silicon blend). "
            "When set to 2, phase-specific parameters ('Primary: ...' / 'Secondary: ...') "
            "must be present in the PyBaMM parameter set."
        ),
    )
    positive_electrode_phases: Literal[1, 2] = Field(
        1,
        description=(
            "Number of particle phases in the positive electrode. "
            "1 = single-phase (default). 2 = composite. "
            "When set to 2, phase-specific parameters ('Primary: ...' / 'Secondary: ...') "
            "must be present in the PyBaMM parameter set."
        ),
    )

    timeseries_rdp_epsilon: float = Field(
        1e-3,
        ge=0,
        description="RDP algorithm tolerance for subsampling in timeseries output",
    )
    experiments: list[ExperimentConfig] | None = Field(
        None,
        description=(
            "Optional list of multi-step / multi-cycle PyBaMM experiments. "
            "Each experiment produces an ExperimentOutput keyed by its label in "
            "experiment_results. Independent of perform_rpt."
        ),
    )
    enable_double_layer_capacitance: bool = Field(
        False,
        description=(
            "Enable double-layer capacitance (C_dl) in the SPMe model. "
            "Adds a non-Faradaic capacitive current term to the total interfacial current: "
            "j_total = j_BV + C_dl * d(phi_s - phi_e)/dt. "
            "Required for hybrid lithium-ion capacitor (LIC) simulations, e.g. "
            "activated-carbon EDLC positive + hard-carbon intercalation negative. "
            "Set per-electrode C_dl via positive/negative_electrode_double_layer_capacitance_F_m2. "
            "PyBaMM default C_dl is 0.2 F/m² when this is enabled and no override is provided."
        ),
    )
    positive_electrode_double_layer_capacitance_F_m2: float | None = Field(
        None,
        gt=0,
        description=(
            "Double-layer capacitance of the positive electrode [F.m-2]. "
            "Only used when enable_double_layer_capacitance=True. "
            "Typical values: activated carbon EDLC ~0.1-0.4 F/m2, "
            "intercalation electrode ~0.01-0.05 F/m2. "
            "Defaults to the PyBaMM built-in value (~0.2 F/m²) when not set."
        ),
    )
    negative_electrode_double_layer_capacitance_F_m2: float | None = Field(
        None,
        gt=0,
        description=(
            "Double-layer capacitance of the negative electrode [F.m-2]. "
            "Only used when enable_double_layer_capacitance=True. "
            "Typical values: hard carbon ~0.01-0.05 F/m2, "
            "graphite ~0.01 F/m2, activated carbon EDLC ~0.1-0.4 F/m2. "
            "Defaults to the PyBaMM built-in value (~0.2 F/m²) when not set."
        ),
    )
    working_electrode: Literal["positive"] | None = Field(
        None,
        description=(
            "Optional working electrode for half-cell simulation. "
            "Set to 'positive' to simulate a half-cell with the positive electrode "
            "as the working electrode against a counter/reference electrode. "
            "The counter electrode defaults to lithium metal (OCP=0 V); use the "
            "reference_electrode_* fields to configure any other counter electrode "
            "(Na metal, pseudo-reference, Li alloy, etc.). "
            "To simulate a negative electrode half-cell (e.g. graphite vs counter), "
            "load the negative electrode material parameters into the positive electrode "
            "slots and set this to 'positive'. "
            "None (default) runs a full-cell simulation. "
            "Incompatible with perform_rpt=True."
        ),
    )
    reference_electrode_exchange_current_density_A_m2: float | None = Field(
        None,
        gt=0,
        description=(
            "Exchange-current density for the counter/reference electrode [A.m-2]. "
            "Only used in half-cell mode (working_electrode='positive'). "
            "Applies to any counter electrode — Li metal, Na metal, pseudo-reference, etc. "
            "When None (default) the Xu2019 Li-metal function is used as a baseline: "
            "j0 = 3.5e-8 * F * c_Li^0.7 * c_e^0.3. "
            "Set to a positive scalar to override with a constant value appropriate "
            "for your counter electrode material."
        ),
    )
    reference_electrode_ocp_V: float | None = Field(
        None,
        description=(
            "Open-circuit potential of the counter/reference electrode [V]. "
            "Only used in half-cell mode (working_electrode='positive'). "
            "Applies to any counter electrode — Li metal (0.0 V), Na metal (~0.3 V vs Li/Li+), "
            "pseudo-reference, lithium alloy, etc. "
            "Defaults to 0.0 V. Set to the known equilibrium potential of your counter electrode."
        ),
    )
    reference_electrode_charge_transfer_coefficient: float | None = Field(
        None,
        gt=0,
        lt=1,
        description=(
            "Butler-Volmer charge transfer coefficient for the counter/reference "
            "electrode (dimensionless, 0 < alpha < 1). "
            "Only used in half-cell mode (working_electrode='positive'). "
            "Applies to any counter electrode material. "
            "Defaults to 0.5 (symmetric kinetics). "
            "Controls the asymmetry of the anodic and cathodic reaction rates."
        ),
    )

    @model_validator(mode="after")
    def _validate_rpt_params_non_null(self) -> SimulationParameters:
        if self.perform_rpt:
            if self.rpt_dcir is None:
                raise ValueError("rpt_dcir must not be None when perform_rpt=True")
            if self.rpt_power is None:
                raise ValueError("rpt_power must not be None when perform_rpt=True")
        return self

    @model_validator(mode="after")
    def _validate_experiment_labels_unique(self) -> SimulationParameters:
        exps = self.experiments
        if not exps:
            return self
        seen: set[str] = set()
        duplicates: list[str] = []
        for exp in exps:
            if exp.label in seen:
                duplicates.append(repr(exp.label))
            else:
                seen.add(exp.label)
        if duplicates:
            raise ValueError(
                "Duplicate experiment labels detected; labels must be unique. "
                f"Conflicts: {', '.join(duplicates)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_double_layer_capacitance_params(self) -> SimulationParameters:
        if not self.enable_double_layer_capacitance:
            if self.positive_electrode_double_layer_capacitance_F_m2 is not None:
                raise ValueError(
                    "positive_electrode_double_layer_capacitance_F_m2 requires "
                    "enable_double_layer_capacitance=True"
                )
            if self.negative_electrode_double_layer_capacitance_F_m2 is not None:
                raise ValueError(
                    "negative_electrode_double_layer_capacitance_F_m2 requires "
                    "enable_double_layer_capacitance=True"
                )
        return self

    @model_validator(mode="after")
    def _validate_half_cell_incompatibilities(self) -> SimulationParameters:
        if self.working_electrode is not None and self.perform_rpt:
            raise ValueError(
                "working_electrode is incompatible with perform_rpt=True; "
                "the RPT uses anode-potential safety cutoffs that are not available "
                "in half-cell mode."
            )
        ref_fields = [
            self.reference_electrode_exchange_current_density_A_m2,
            self.reference_electrode_ocp_V,
            self.reference_electrode_charge_transfer_coefficient,
        ]
        if any(v is not None for v in ref_fields) and self.working_electrode is None:
            raise ValueError(
                "reference_electrode_* fields are only used in half-cell mode "
                "(working_electrode='positive') and would be silently ignored in a "
                "full-cell simulation. Set working_electrode='positive' or remove "
                "the reference_electrode_* overrides."
            )
        return self


class CellPerformanceInput(BaseModel):
    cell_parameters: CellParametersInput
    simulation_parameters: SimulationParameters

    @model_validator(mode="after")
    def _validate_volume_dimensions(self) -> CellPerformanceInput:
        sp = self.simulation_parameters
        # Volume dimensions are only required when PyBaMM will actually run
        # (RPT enabled or at least one experiment configured).
        needs_simulation = sp.perform_rpt or bool(sp.experiments)
        if not needs_simulation:
            return self
        cp = self.cell_parameters
        ff = cp.form_factor.lower()
        if ff in ("coin", "cylindrical"):
            missing = []
            if cp.cell_diameter_mm is None:
                missing.append("cell_diameter_mm")
            elif cp.cell_diameter_mm <= 0:
                raise ValueError(
                    f"cell_diameter_mm must be > 0, got {cp.cell_diameter_mm}."
                )
            if cp.cell_height_mm is None:
                label = "coin thickness" if ff == "coin" else "cylinder height"
                missing.append(f"cell_height_mm ({label})")
            elif cp.cell_height_mm <= 0:
                raise ValueError(
                    f"cell_height_mm must be > 0, got {cp.cell_height_mm}."
                )
            if missing:
                raise ValueError(
                    f"{cp.form_factor} cell is missing {' and '.join(missing)}. "
                    "Both are required to compute cell volume for PyBaMM simulation."
                )
        elif ff in ("pouch", "prismatic"):
            missing = []
            if cp.cell_width_mm is None:
                missing.append("cell_width_mm")
            elif cp.cell_width_mm <= 0:
                raise ValueError(f"cell_width_mm must be > 0, got {cp.cell_width_mm}.")
            if cp.cell_height_mm is None:
                missing.append("cell_height_mm")
            elif cp.cell_height_mm <= 0:
                raise ValueError(
                    f"cell_height_mm must be > 0, got {cp.cell_height_mm}."
                )
            if cp.cell_thickness_mm is None:
                missing.append("cell_thickness_mm")
            elif cp.cell_thickness_mm <= 0:
                raise ValueError(
                    f"cell_thickness_mm must be > 0, got {cp.cell_thickness_mm}."
                )
            if missing:
                raise ValueError(
                    f"{cp.form_factor} cell is missing {', '.join(missing)}. "
                    "All three are required to compute cell volume for PyBaMM simulation."
                )
        return self


# --- Output schemas ---
class TimeseriesOutput(BaseModel):
    time_s: list[
        float
    ]  # Seconds from simulation start (experiment: accumulated across all steps/cycles; C/3 discharge: from step start)
    voltage_V: list[float]
    current_A: list[float]
    temperature_K: list[float]
    capacity_Ah: list[float]
    energy_Wh: list[float]
    power_W: list[float]
    anode_potential_V: list[float]
    # Populated for CCCV charge tests only.
    cc_capacity_Ah: float | None = None
    total_capacity_Ah: float | None = None


class CellEquilibriumKPIs(BaseModel):
    """Equilibrium outputs from electrode balancing and initial stoichiometry calculation."""

    model_config = ConfigDict(extra="forbid")

    # Equilibrium
    cell_theoretical_capacity_Ah: float = Field(
        ..., description="Theoretical cell capacity [Ah]"
    )
    cell_theoretical_energy_Wh: float = Field(
        ..., description="Theoretical cell energy [Wh]"
    )
    cell_n_p_ratio: float = Field(..., description="Cell N:P ratio")
    stoichiometry_windows: StoichiometryWindows = Field(
        ...,
        description="Electrode stoichiometry windows from equilibrium + coulombic loss",
    )
    cell_mass_g: float = Field(..., description="Cell mass [g]")
    cell_volume_L: float = Field(..., description="Cell volume [L]")

    cell_theoretical_gravimetric_energy_density_Wh_kg: float = Field(
        ..., description="Theoretical gravimetric energy density [Wh/kg]"
    )
    cell_theoretical_volumetric_energy_density_Wh_L: float = Field(
        ..., description="Theoretical volumetric energy density [Wh/L]"
    )
    positive_electrode_porosity: float = Field(
        ..., ge=0, le=1, description="Positive electrode porosity"
    )
    negative_electrode_porosity: float = Field(
        ..., ge=0, le=1, description="Negative electrode porosity"
    )
    positive_electrode_density_g_cm3: float = Field(
        ..., description="Positive electrode density [g/cm³]"
    )
    negative_electrode_density_g_cm3: float = Field(
        ..., description="Negative electrode density [g/cm³]"
    )
    positive_electrode_volume_fraction: float = Field(
        ..., ge=0, le=1, description="Positive electrode volume fraction in cell"
    )
    negative_electrode_volume_fraction: float = Field(
        ..., ge=0, le=1, description="Negative electrode volume fraction in cell"
    )
    electrolyte_mass_g: float = Field(..., description="Electrolyte mass [g]")
    electrolyte_mass_fraction: float = Field(
        ..., ge=0, le=1, description="Electrolyte mass fraction"
    )
    jelly_roll_mass_g: float = Field(..., description="Jelly roll mass [g]")
    jelly_roll_mass_fraction: float = Field(
        ..., ge=0, le=1, description="Jelly roll mass fraction"
    )
    casing_mass_g: float = Field(..., description="Casing mass [g]")
    casing_mass_fraction: float = Field(
        ..., ge=0, le=1, description="Casing mass fraction"
    )
    bill_of_materials: BillOfMaterials | None = Field(
        None, description="Bill of materials with mass breakdown by component"
    )


class CellPerformanceKPIs(CellEquilibriumKPIs):
    """Output equilibrium + PyBaMM KPIs."""

    model_config = ConfigDict(extra="forbid")

    # PyBaMM

    cell_nominal_capacity_Ah: float | None = Field(
        None, description="Cell nominal capacity at specified C-rate [Ah]"
    )
    cell_nominal_energy_Wh: float | None = Field(
        None, description="Cell nominal energy [Wh]"
    )
    cell_nominal_gravimetric_energy_density_Wh_kg: float | None = Field(
        None, description="Cell nominal gravimetric energy density [Wh/kg]"
    )
    cell_nominal_volumetric_energy_density_Wh_L: float | None = Field(
        None, description="Cell nominal volumetric energy density [Wh/L]"
    )
    cell_nominal_max_discharge_power_300s_W: float | None = Field(
        None, description="Cell nominal max discharge power (300s) [W]"
    )
    cell_nominal_gravimetric_power_density_W_kg: float | None = Field(
        None, description="Cell nominal gravimetric power density [W/kg]"
    )
    cell_nominal_volumetric_power_density_W_L: float | None = Field(
        None, description="Cell nominal volumetric power density [W/L]"
    )
    cell_dcir_10s_mohm: float | None = Field(None, description="Cell DCIR (10s) [mΩ]")
    ocv100_V: float | None = Field(
        None, description="Open-circuit voltage at 100% SOC [V]"
    )
    ocv0_V: float | None = Field(None, description="Open-circuit voltage at 0% SOC [V]")
    c3_discharge_timeseries: TimeseriesOutput | None = Field(
        None, description="C/3 discharge time-series data (subsampled)"
    )
    experiment_results: dict[str, ExperimentOutput] | None = Field(
        None,
        description=(
            "Multi-cycle experiment results keyed by ExperimentConfig.label. "
            "Present only when experiments were requested."
        ),
    )


# =============================================================================
# SECTION 2 — GEOMETRY & BILL-OF-MATERIALS HELPERS
# =============================================================================


def _inv_rho(ams: list, binders: list, conds: list) -> float:
    """Sum of mass_fraction/density for all formulation components."""
    return sum(x.mass_fraction / x.density_g_cm3 for x in [*ams, *binders, *conds])


def _formulation_density_g_cm3(
    active_materials: list[ActiveMaterial],
    binders: list[Binder],
    conductive_agents: list[ConductiveAgent],
) -> float:
    ir = _inv_rho(active_materials, binders, conductive_agents)
    return 1.0 / ir if ir > 0 else 0.0


def _electrode_volume_fractions(
    porosity: float,
    active_materials: list[ActiveMaterial],
    binders: list[Binder],
    conductive_agents: list[ConductiveAgent],
) -> tuple[float, float, float]:
    """(am_vol_frac, binder_vol_frac, cond_vol_frac) using shared inverse-density."""
    ir = _inv_rho(active_materials, binders, conductive_agents)
    if ir <= 0:
        return 0.0, 0.0, 0.0
    solid = 1.0 - porosity
    am_vf = (
        solid * sum(am.mass_fraction / am.density_g_cm3 for am in active_materials) / ir
    )
    bi_vf = solid * sum(b.mass_fraction / b.density_g_cm3 for b in binders) / ir
    co_vf = (
        solid * sum(c.mass_fraction / c.density_g_cm3 for c in conductive_agents) / ir
    )
    return am_vf, bi_vf, co_vf


def _capacity_weighted_voltage_V(active_materials: list[ActiveMaterial]) -> float:
    """Capacity-weighted average voltage for (possibly blended) electrode.

    Each material contributes capacity proportional to:
        mass_fraction x specific_capacity_mAh_g
    The voltage is then the capacity-weighted mean of nominal_voltage_V.

    This is physically correct for blends (e.g. NMC+LFP cathode, Gr+SiOx anode)
    because it weights by how much charge each material delivers, not by mass alone.
    For single-material electrodes, this reduces to that material's nominal_voltage_V.
    """
    total_cap_weight = 0.0
    weighted_voltage = 0.0
    for am in active_materials:
        cap_contribution = am.mass_fraction * am.specific_capacity_mAh_g
        weighted_voltage += cap_contribution * am.nominal_voltage_V
        total_cap_weight += cap_contribution
    if total_cap_weight <= 0:
        return 0.0
    return weighted_voltage / total_cap_weight


def _jelly_roll_geometry_cylindrical(
    cell_params: CellParametersInput,
) -> tuple[float, float, float, float]:
    jr_outer = cell_params.cell_diameter_mm - 2 * cell_params.casing_thickness_mm - 0.1
    jr_inner = cell_params.jelly_roll_inner_diameter_mm
    pos_um = (
        2 * cell_params.positive_coating_thickness_um
        + cell_params.positive_electrode_foil_thickness_um
    )
    neg_um = (
        2 * cell_params.negative_coating_thickness_um
        + cell_params.negative_electrode_foil_thickness_um
    )
    sep_um = 2 * cell_params.separator_thickness_um
    layer_mm = (pos_um + neg_um + sep_um) * 1e-3
    if layer_mm <= 0:
        return jr_outer, jr_inner, 0.0, 0.0
    rad_diff = jr_outer / 2 - jr_inner / 2
    if rad_diff <= 0:
        return jr_outer, jr_inner, 0.0, 0.0
    windings = int(rad_diff / layer_mm)
    pos_width = (jr_outer**2 - jr_inner**2) * pi / 4 / layer_mm if windings > 0 else 0.0
    return jr_outer, jr_inner, float(windings), pos_width


# Electrolyte
def _coat_vol(
    electrode_coating_side_count,
    jelly_roll_count,
    sheet_count,
    electrode_area_cm2,
    thickness_um,
):
    return (
        electrode_coating_side_count
        * jelly_roll_count
        * sheet_count
        * electrode_area_cm2
        * (thickness_um * 1e-4)
    )


def _coat_mass(
    electrode_coating_side_count,
    jelly_roll_count,
    sheet_count,
    electrode_area_cm2,
    mass_loading,
) -> float:
    return (
        electrode_coating_side_count
        * jelly_roll_count
        * sheet_count
        * electrode_area_cm2
        * (mass_loading * 1e-3)
    )


def _foil_mass(
    electrode_area_cm2, foil_thickness_um, foil_density, jelly_roll_count, sheet_count
) -> float:
    return (
        jelly_roll_count
        * sheet_count
        * electrode_area_cm2
        * (foil_thickness_um * 1e-4)
        * foil_density
    )


def _electrode_area_cm2(
    cell_params: CellParametersInput, is_positive: bool = True
) -> float:
    # If electrode dimensions are provided directly, use them
    ff = cell_params.form_factor.lower()

    # Check electrode-specific dimensions first
    if (
        is_positive
        and cell_params.positive_electrode_width_mm is not None
        and cell_params.positive_electrode_height_mm is not None
    ):
        width = cell_params.positive_electrode_width_mm
        height = cell_params.positive_electrode_height_mm
        return (width * height) / 100.0
    if (
        not is_positive
        and cell_params.negative_electrode_width_mm is not None
        and cell_params.negative_electrode_height_mm is not None
    ):
        width = cell_params.negative_electrode_width_mm
        height = cell_params.negative_electrode_height_mm
        return (width * height) / 100.0
    if ff == "cylindrical":
        _jr_outer, _jr_inner, _windings, pos_width = _jelly_roll_geometry_cylindrical(
            cell_params
        )
        if pos_width <= 0:
            return 0.0
        sep_h = cell_params.cell_height_mm * cell_params.volume_packing_ratio
        pos_h = max(0.0, sep_h - 4 * cell_params.electrode_overhang_mm)
        neg_h = max(0.0, sep_h - 2 * cell_params.electrode_overhang_mm)
        neg_width = max(0.0, pos_width + 2 * cell_params.electrode_overhang_mm)
        if is_positive:
            return pos_width * pos_h / 100.0
        else:
            return neg_width * neg_h / 100.0
    elif ff == "coin":
        if not cell_params.cell_diameter_mm:
            return 0.0
        oh = cell_params.electrode_overhang_mm
        pos_dia = max(0.0, cell_params.cell_diameter_mm - 4 * oh)
        neg_dia = max(0.0, cell_params.cell_diameter_mm - 2 * oh)
        if is_positive:
            return np.pi * (pos_dia / 2) ** 2 / 100.0
        else:
            return np.pi * (neg_dia / 2) ** 2 / 100.0
    elif ff in ("pouch", "prismatic"):
        if not cell_params.cell_width_mm or not cell_params.cell_height_mm:
            return 0.0
        oh = cell_params.electrode_overhang_mm
        vp = cell_params.volume_packing_ratio
        pos_w = max(0.0, cell_params.cell_width_mm * vp - 4 * oh)
        pos_h = max(0.0, cell_params.cell_height_mm * vp - 4 * oh)
        return pos_w * pos_h / 100.0
    else:
        return 0.0


def _cell_volume_L(cell_params: CellParametersInput) -> float:
    if (
        cell_params.form_factor.lower() == "cylindrical"
        and cell_params.cell_diameter_mm
        and cell_params.cell_height_mm
    ):
        return (
            pi
            * (cell_params.cell_diameter_mm / 2) ** 2
            * cell_params.cell_height_mm
            / 1e6
        )
    if (
        cell_params.form_factor.lower() in ("pouch", "prismatic")
        and cell_params.cell_width_mm
        and cell_params.cell_height_mm
        and cell_params.cell_thickness_mm
    ):
        return (
            cell_params.cell_width_mm
            * cell_params.cell_height_mm
            * cell_params.cell_thickness_mm
            / 1e6
        )
    if (
        cell_params.form_factor.lower() == "coin"
        and cell_params.cell_diameter_mm
        and cell_params.cell_height_mm
    ):
        return (
            np.pi
            * (cell_params.cell_diameter_mm / 2) ** 2
            * cell_params.cell_height_mm
            / 1e6
        )
    return 0.0


def item(
    name: str,
    electrode: str | None,
    mass_g: float,
    cell_mass_g: float,
    tol: float = 1e-9,
) -> BillOfMaterialsItem | None:
    """Create a BillOfMaterialsItem if mass exceeds tolerance, otherwise return None.

    NOTE: This function intentionally returns None for zero-mass components (below tolerance).
    Callers use truthiness filtering (e.g., `if (it := item(...))`) to silently drop these
    components from the BOM. This prevents zero-mass entries from rounding errors or
    calculation artifacts from appearing in the bill of materials.

    Args:
        name: Material name
        electrode: Electrode type ("positive", "negative", or None)
        mass_g: Component mass [g]
        cell_mass_g: Total cell mass [g]
        tol: Mass tolerance threshold [g]. Components with mass < tol return None.

    Returns:
        BillOfMaterialsItem if mass >= tol, None otherwise
    """
    if mass_g < tol:
        return None
    frac = mass_g / cell_mass_g if cell_mass_g > 0 else 0.0
    return BillOfMaterialsItem(
        name=name,
        electrode=electrode,
        mass_g=round(mass_g, 4),
        mass_fraction=round(frac, 4),
    )


def _compute_bill_of_materials(
    cell_params: CellParametersInput,
    cell_mass_g: float,
    electrolyte_mass_g: float,
    casing_mass_g: float,
) -> BillOfMaterials:
    tol = 1e-9
    positive_electrode_area_cm2 = _electrode_area_cm2(cell_params, is_positive=True)
    negative_electrode_area_cm2 = _electrode_area_cm2(cell_params, is_positive=False)
    pos_coat = _coat_mass(
        cell_params.electrode_coating_side_count,
        cell_params.jelly_roll_count,
        cell_params.positive_electrode_sheet_count,
        positive_electrode_area_cm2,
        cell_params.positive_electrode_mass_loading_mg_cm2,
    )
    neg_coat = _coat_mass(
        cell_params.electrode_coating_side_count,
        cell_params.jelly_roll_count,
        cell_params.negative_electrode_sheet_count,
        negative_electrode_area_cm2,
        cell_params.negative_electrode_mass_loading_mg_cm2,
    )
    sep_sheet_count = cell_params.separator_sheet_count
    sep_area_cm2 = (
        cell_params.separator_sheet_width_mm
        * cell_params.separator_sheet_height_mm
        / 100.0
    )
    # Separator mass: jelly_roll_count * sep_layers * area * thickness * density * (1 - porosity)
    # Match the equilibrium calculation at line ~1919
    sep_vol_cm3 = (
        sep_sheet_count * sep_area_cm2 * (cell_params.separator_thickness_um * 1e-4)
    )
    sep_mass = (
        cell_params.jelly_roll_count
        * sep_vol_cm3
        * cell_params.separator_density_g_cm3
        * (1 - cell_params.separator_porosity)
    )
    pos_foil = _foil_mass(
        positive_electrode_area_cm2,
        cell_params.positive_electrode_foil_thickness_um,
        cell_params.positive_electrode_foil_density_g_cm3,
        cell_params.jelly_roll_count,
        cell_params.positive_electrode_sheet_count,
    )
    neg_foil = _foil_mass(
        negative_electrode_area_cm2,
        cell_params.negative_electrode_foil_thickness_um,
        cell_params.negative_electrode_foil_density_g_cm3,
        cell_params.jelly_roll_count,
        cell_params.negative_electrode_sheet_count,
    )
    active = [
        it
        for am in cell_params.positive_electrode_active_materials
        if (
            it := item(
                am.name, "positive", pos_coat * am.mass_fraction, cell_mass_g, tol
            )
        )
    ]
    active += [
        it
        for am in cell_params.negative_electrode_active_materials
        if (
            it := item(
                am.name, "negative", neg_coat * am.mass_fraction, cell_mass_g, tol
            )
        )
    ]
    binders = [
        it
        for b in cell_params.positive_electrode_binders
        if (
            it := item(b.name, "positive", pos_coat * b.mass_fraction, cell_mass_g, tol)
        )
    ]
    binders += [
        it
        for b in cell_params.negative_electrode_binders
        if (
            it := item(b.name, "negative", neg_coat * b.mass_fraction, cell_mass_g, tol)
        )
    ]
    conductive = [
        it
        for c in cell_params.positive_electrode_conductive_agents
        if (
            it := item(c.name, "positive", pos_coat * c.mass_fraction, cell_mass_g, tol)
        )
    ]
    conductive += [
        it
        for c in cell_params.negative_electrode_conductive_agents
        if (
            it := item(c.name, "negative", neg_coat * c.mass_fraction, cell_mass_g, tol)
        )
    ]

    foils = [
        BillOfMaterialsItem(
            name=name,
            electrode=elec,
            mass_g=round(mass, 4),
            mass_fraction=round(mass / cell_mass_g, 4) if cell_mass_g > 0 else 0,
        )
        for name, elec, mass in [
            ("Al foil", "positive", pos_foil),
            ("Cu foil", "negative", neg_foil),
        ]
        if mass >= tol
    ]

    return BillOfMaterials(
        active_materials=active,
        binders=binders,
        conductive_agents=conductive,
        separator=item("separator", None, sep_mass, cell_mass_g, tol),
        foils=foils,
        electrolyte=item("electrolyte", None, electrolyte_mass_g, cell_mass_g, tol),
        casing=item("casing", None, casing_mass_g, cell_mass_g, tol),
    )


# =============================================================================
# SECTION 3 — PyBaMM PARAMETER BUILDING
# =============================================================================


def _first_material_with(
    materials: list[ActiveMaterial], attr: str
) -> ActiveMaterial | None:
    return next((m for m in materials if getattr(m, attr, None) is not None), None)


def _rdp_subsample(data: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker algorithm for subsampling 2D data (N, M).

    Note: This implementation uses only the first two columns (time, voltage) for
    distance calculation, but returns all columns. This means large excursions in
    other signals (current, temperature, anode potential) do not contribute to
    point selection. A significant temperature spike or current transient between
    two time points that happen to lie on the voltage curve could be dropped.

    This is acceptable for the primary use case (voltage-time curve subsampling)
    but should be considered when interpreting subsampled data with important
    secondary signals.

    Args:
        data: Array of shape (N, M) where first column is time, second is voltage,
              and remaining columns are other signals (current, temperature, etc.)
        epsilon: Maximum perpendicular distance threshold for point retention

    Returns:
        Subsampled array preserving all columns
    """
    if data.shape[0] <= 2:
        return data

    # Use first two columns (time, voltage) for distance calculation
    # but we stack everything for return.
    # LIMITATION: Other signals (current, temperature, anode potential) do not
    # contribute to point selection, only voltage-time shape is preserved.
    d = data[:, :2]
    start = d[0, :]
    end = d[-1, :]

    # Line vector from start to end
    line_vec = end - start
    line_len_sq = np.sum(line_vec**2)

    if line_len_sq == 0:
        # All points are at the same location?
        dist_sq = np.sum((d - start) ** 2, axis=1)
    else:
        # Projection of all points onto line
        t = np.sum((d - start) * line_vec, axis=1) / line_len_sq
        t = np.clip(t, 0.0, 1.0)
        projection = start + t[:, np.newaxis] * line_vec
        dist_sq = np.sum((d - projection) ** 2, axis=1)

    idx = np.argmax(dist_sq)
    max_dist = np.sqrt(dist_sq[idx])

    if max_dist > epsilon:
        # If the max distance point is one of the ends, we stop to avoid infinite recursion
        if idx == 0 or idx == data.shape[0] - 1:
            return data[[0, -1], :]

        left = _rdp_subsample(data[: idx + 1, :], epsilon)
        right = _rdp_subsample(data[idx:, :], epsilon)
        return np.vstack((left[:-1, :], right))
    else:
        return data[[0, -1], :]


def _to_pybamm_callable(table: IonicPropertyTable) -> Any:
    """Convert IonicPropertyTable to PyBaMM callable function.

    Returns a function that interpolates over concentration and temperature.
    When PyBaMM calls this with symbolic arguments during model building, it
    returns a scalar fallback value evaluated at reference conditions (mean
    concentration and mean temperature). This scalar is wrapped as a constant
    in the expression tree, effectively disabling concentration and temperature
    dependence during symbolic evaluation. At runtime, when numeric values are
    provided, full interpolation is performed.

    LIMITATION: The fallback behavior means that concentration and temperature
    dependence is lost during model building. This is acceptable for properties
    that vary weakly with these parameters, but may introduce errors for strongly
    dependent properties (e.g., ionic conductivity at low concentrations).
    """
    c = np.array(table.conc_mol_m3, dtype=float)
    T = np.array(table.temp_K, dtype=float)
    v = np.array(table.value, dtype=float)
    if len(c) < 3:
        const = float(np.mean(v))
        return lambda c_e, T_k: const  # noqa: ARG005
    points = np.column_stack([c, T])
    interp = LinearNDInterpolator(points, v, fill_value=np.nan)

    # Use reference conditions (mean concentration and temperature) for fallback
    # instead of just mean value, to provide better-conditioned constant
    # when PyBaMM evaluates symbolically
    ref_c = float(np.nanmean(c))
    ref_T = float(np.nanmean(T))
    ref_val = interp(np.atleast_1d(ref_c), np.atleast_1d(ref_T))
    fallback = (
        float(ref_val.flat[0])
        if ref_val.size > 0 and not np.isnan(ref_val.flat[0])
        else float(np.nanmean(v))
    )

    def _is_numeric(x) -> bool:
        """Check if x is a real number (not a PyBaMM symbolic object)."""
        try:
            float(x)
            return True
        except (TypeError, ValueError):
            return False

    def f(c_e, T_k):
        # PyBaMM calls FunctionParameter callables with symbolic objects
        # (pybamm.Maximum, pybamm.StateVector, etc.) during model building.
        # We cannot interpolate over symbolic inputs, so return the scalar
        # fallback (value at reference conditions: mean concentration and temperature).
        # PyBaMM will wrap this as a Scalar node in the expression tree.
        # LIMITATION: This disables concentration and temperature dependence
        # during symbolic evaluation, but full interpolation is used at runtime.
        if not (_is_numeric(c_e) and _is_numeric(T_k)):
            return fallback

        out = interp(np.atleast_1d(c_e), np.atleast_1d(T_k))
        if np.isscalar(c_e) and np.isscalar(T_k):
            val = float(out.flat[0]) if out.size > 0 else np.nan
            return fallback if np.isnan(val) else val
        return np.where(np.isnan(out), fallback, out)

    return f


def _electrolyte_param_overrides(cell_params: CellParametersInput) -> dict[str, Any]:
    if cell_params.electrolyte is None:
        return {}
    e = cell_params.electrolyte
    out = {}
    mapping = [
        (e.ionic_conductivity_S_m, "Electrolyte conductivity [S.m-1]"),
        (e.ionic_diffusivity_m2_s, "Electrolyte diffusivity [m2.s-1]"),
        (e.transference_number, "Cation transference number"),
        (e.activity_coefficient, "Thermodynamic factor"),
    ]
    for val, key in mapping:
        if val is not None:
            out[key] = (
                float(val) if isinstance(val, int | float) else _to_pybamm_callable(val)
            )
    return out


def _active_material_param_overrides(
    cell_params: CellParametersInput,
) -> dict[str, Any]:
    out = {}
    for materials, prefix in [
        (cell_params.positive_electrode_active_materials, "Positive"),
        (cell_params.negative_electrode_active_materials, "Negative"),
    ]:
        mat_diff = _first_material_with(materials, "diffusivity_m2_s")
        if mat_diff and mat_diff.diffusivity_m2_s is not None:
            d = mat_diff.diffusivity_m2_s
            key = f"{prefix} particle diffusivity [m2.s-1]"
            out[key] = (
                float(d) if isinstance(d, int | float) else _to_pybamm_callable(d)
            )

        mat_psd = _first_material_with(materials, "particle_size_distribution")
        if mat_psd and mat_psd.particle_size_distribution is not None:
            out[f"{prefix} particle radius [m]"] = (
                mat_psd.particle_size_distribution.radius_m()
            )

        mat_j0 = _first_material_with(materials, "exchange_current_density_A_m2")
        if mat_j0 and mat_j0.exchange_current_density_A_m2 is not None:
            out[f"{prefix} electrode exchange-current density [A.m-2]"] = float(
                mat_j0.exchange_current_density_A_m2
            )

        mat_cdl = _first_material_with(materials, "double_layer_capacitance_F_m2")
        if mat_cdl and mat_cdl.double_layer_capacitance_F_m2 is not None:
            out[f"{prefix} electrode double-layer capacity [F.m-2]"] = float(
                mat_cdl.double_layer_capacitance_F_m2
            )
    return out


def _tortuosity_param_overrides(
    cell_params: CellParametersInput,
) -> dict[str, float]:
    """Tortuosity factor overrides for PyBaMM (electrolyte + solid phases)."""
    out: dict[str, float] = {}
    for prefix, pore, solid in [
        (
            "Positive",
            cell_params.positive_electrode_pore_tortuosity,
            cell_params.positive_electrode_solid_tortuosity,
        ),
        (
            "Negative",
            cell_params.negative_electrode_pore_tortuosity,
            cell_params.negative_electrode_solid_tortuosity,
        ),
    ]:
        if pore is not None:
            ev = float(pore)
            sv = float(solid) if solid is not None else ev
            out[f"{prefix} electrode tortuosity factor (electrolyte)"] = ev
            out[f"{prefix} electrode tortuosity factor (electrode)"] = sv
    if cell_params.separator_tortuosity is not None:
        out["Separator tortuosity factor (electrolyte)"] = float(
            cell_params.separator_tortuosity
        )
    return out


def _get_spme_model_options(
    cell_params: CellParametersInput,
    thermal: str = "isothermal",
    working_electrode: str | None = None,
    particle_diffusion_model: str = "Fickian diffusion",
    negative_electrode_phases: int = 1,
    positive_electrode_phases: int = 1,
    enable_double_layer_capacitance: bool = False,
) -> dict[str, object]:
    options: dict[str, object] = {
        "calculate discharge energy": "true",
        "cell geometry": "arbitrary",
        "thermal": thermal,
        "contact resistance": "true",
        "SEI": "none",
        "particle mechanics": "none",
        "loss of active material": "none",
        "particle": particle_diffusion_model,
    }
    if any(
        [
            cell_params.positive_electrode_pore_tortuosity is not None,
            cell_params.negative_electrode_pore_tortuosity is not None,
            cell_params.separator_tortuosity is not None,
        ]
    ):
        options["transport efficiency"] = "tortuosity factor"
    if working_electrode is not None:
        options["working electrode"] = working_electrode
    if enable_double_layer_capacitance:
        options["double-layer capacitance"] = "true"
    if negative_electrode_phases > 1 or positive_electrode_phases > 1:
        # PyBaMM expects (negative_phases, positive_phases) as string digits.
        options["particle phases"] = (
            str(negative_electrode_phases),
            str(positive_electrode_phases),
        )

        # open-circuit potential must be shaped to match phases per electrode.
        # "single" means OCP comes from the parameter-defined function for each phase.
        def _ocp_for_phases(n: int) -> object:
            return tuple("single" for _ in range(n)) if n > 1 else "single"

        options["open-circuit potential"] = (
            _ocp_for_phases(negative_electrode_phases),
            _ocp_for_phases(positive_electrode_phases),
        )
    return options


def _make_spme_model(
    cell_params: CellParametersInput,
    sim_params: SimulationParameters,
    thermal: str = "isothermal",
    *,
    set_anode_alias: bool = True,
) -> Any:
    """Create a configured SPMe model from sim_params options.

    When set_anode_alias=True (default) and not in positive half-cell mode, adds the
    'Anode potential [V]' variable alias required by downstream extraction functions.
    Pass set_anode_alias=False for simulations that do not need the anode potential
    output (e.g. calibration, DCIR).
    """
    options = _get_spme_model_options(
        cell_params,
        thermal=thermal,
        working_electrode=sim_params.working_electrode,
        particle_diffusion_model=sim_params.particle_diffusion_model,
        negative_electrode_phases=sim_params.negative_electrode_phases,
        positive_electrode_phases=sim_params.positive_electrode_phases,
        enable_double_layer_capacitance=sim_params.enable_double_layer_capacitance,
    )
    model = pybamm.lithium_ion.SPMe(options=options)
    if set_anode_alias and sim_params.working_electrode != "positive":
        model.variables["Anode potential [V]"] = model.variables[
            "Negative electrode surface potential difference at separator interface [V]"
        ]
    return model


# --- OCV model helpers ---
@functools.lru_cache(maxsize=1)
def _get_halfcell_spme_defaults() -> dict:
    """Return cached half-cell SPMe default parameter values.

    Building a PyBaMM model just to read its defaults is expensive; the result
    is cached via lru_cache so repeated half-cell runs reuse the same dict.
    """
    import pybamm as _pb

    return dict(
        _pb.lithium_ion.SPMe(
            options={"working electrode": "positive"}
        ).default_parameter_values
    )


@functools.lru_cache(maxsize=1)
def _check_pybammeis() -> bool:
    """Return True if pybammeis is installed, False otherwise (result is cached)."""
    try:
        import pybammeis  # type: ignore[import]  # noqa: F401

        return True
    except ImportError:
        logger.warning(
            "pybammeis not installed - EIS steps will be skipped. "
            "Install pybammeis to enable impedance support."
        )
        return False


def _reset_pybammeis_cache() -> None:
    """Clear the pybammeis availability cache.  Call in test teardown to avoid
    cross-test contamination when monkeypatching import availability."""
    _check_pybammeis.cache_clear()


def _reset_halfcell_spme_defaults_cache() -> None:
    """Clear the half-cell SPMe defaults cache.  Call in test teardown to avoid
    cross-test contamination when monkeypatching PyBaMM model construction."""
    _get_halfcell_spme_defaults.cache_clear()


_CHEM_POLY_DEG: dict[str, int] = {
    "LFP": 18,
    "LFMP": 18,
    "LTO": 16,
    "LMO": 14,
    "NMC": 10,
    "NCA": 10,
    "graphite": 12,
}
_DEFAULT_POLY_DEG = 12


def _poly_deg_for_chemistry(chemistry: str | None) -> int:
    """Return a sensible polynomial degree for the given chemistry string.

    Args:
        chemistry: Material name or chemistry identifier (e.g., "NMC811", "LFP", "Graphite")

    Returns:
        Polynomial degree appropriate for the chemistry, or default if not recognized
    """
    if chemistry is None:
        return _DEFAULT_POLY_DEG
    chem_upper = chemistry.upper()
    for key, deg in _CHEM_POLY_DEG.items():
        if chem_upper.startswith(key.upper()):
            return deg
    return _DEFAULT_POLY_DEG


# --- MSMR fitting helpers ---
_R = 8.314462618  # J/(mol·K)
_F = 96485.3329  # C/mol


def _msmr_x_from_V(V: np.ndarray, params: np.ndarray, T: float) -> np.ndarray:
    """Compute total stoichiometry x(V) from MSMR site parameters: x_j = Xj/(1+exp(F(V-U0j)/(RT*kj)))."""
    n_sites = len(params) // 3
    x = np.zeros_like(V, dtype=float)
    for j in range(n_sites):
        U0j = params[3 * j]
        kj = params[3 * j + 1]
        Xj = params[3 * j + 2]
        arg = _F * (V - U0j) / (_R * T * kj)
        # Clip to avoid overflow
        arg = np.clip(arg, -500.0, 500.0)
        x += Xj / (1.0 + np.exp(arg))
    return x


def _fit_msmr(
    sto: np.ndarray,
    volt: np.ndarray,
    n_sites: int,
    T: float = 298.15,
) -> np.ndarray:
    """Fit MSMR parameters (U0_j, κ_j, X_j) to tabulated OCV data via NLS."""
    V_min, V_max = volt.min(), volt.max()
    x_total = sto[-1] - sto[0]

    # Initial guess: spread U0 evenly across voltage range,
    # equal capacity fractions, moderate shape factors
    p0 = []
    for j in range(n_sites):
        U0_init = V_min + (V_max - V_min) * (j + 0.5) / n_sites
        kappa_init = 2.0
        Xj_init = x_total / n_sites
        p0.extend([U0_init, kappa_init, Xj_init])
    p0 = np.array(p0)

    # Bounds: U0 within voltage range (with margin), kappa > 0, Xj > 0
    # Constraint: Σ Xj ≈ x_total (enforced via penalty in residuals)
    lb = []
    ub = []
    V_margin = 0.3 * (V_max - V_min)
    for _ in range(n_sites):
        lb.extend([V_min - V_margin, 0.1, 1e-4])
        # Upper bound for Xj: allow up to x_total per site, but constraint enforces sum ≈ x_total
        ub.extend([V_max + V_margin, 50.0, x_total])

    def residuals(params):
        x_pred = _msmr_x_from_V(volt, params, T)
        # Extract Xj values (indices 2, 5, 8, ...)
        Xj_values = params[2::3]
        sum_Xj = np.sum(Xj_values)
        # Penalty term: enforce Σ Xj ≈ x_total
        # Weight the constraint penalty relative to data fit (use same scale as data)
        constraint_penalty = (
            10.0 * abs(sum_Xj - x_total) / x_total if x_total > 0 else 0.0
        )
        data_residuals = x_pred - sto
        # Append constraint penalty as additional residual
        return np.append(data_residuals, constraint_penalty)

    result = least_squares(
        residuals,
        p0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=10000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    # Normalize Xj values to ensure exact sum = x_total (post-optimization correction)
    params_fitted = result.x.copy()
    Xj_indices = np.arange(2, len(params_fitted), 3)
    Xj_sum = np.sum(params_fitted[Xj_indices])
    if Xj_sum > 0:
        # Scale all Xj proportionally to sum to x_total
        params_fitted[Xj_indices] = params_fitted[Xj_indices] * (x_total / Xj_sum)

    return params_fitted


# --- OCP builders (interpolant & blend) ---
def _make_ocp_func(
    sto_grid: np.ndarray,
    volt_grid: np.ndarray,
    pb: Any,
    ocp_mode: str,
    poly_deg: int | None = None,
    msmr_n_sites: int | None = None,
    msmr_T: float = 298.15,
) -> Any:
    """Return a PyBaMM-compatible OCP callable from tabulated (sto, V) data."""
    if ocp_mode == "polynomial":
        if poly_deg is None:
            raise ValueError("poly_deg is required for ocp_mode='polynomial'")
        coeffs = np.polyfit(sto_grid, volt_grid, poly_deg)

        def ocp_func(sto):
            result = coeffs[0]
            for c in coeffs[1:]:
                result = result * sto + c
            return result

    elif ocp_mode == "msmr":
        if msmr_n_sites is None:
            raise ValueError("msmr_n_sites is required for ocp_mode='msmr'")

        params = _fit_msmr(sto_grid, volt_grid, msmr_n_sites, T=msmr_T)

        # MSMR defines x(V) implicitly, but PyBaMM needs V(sto).
        # We invert numerically: build a fine V(x) table from the fit,
        # then return a polynomial fit of V(sto) from that table.
        # This preserves the thermodynamic shape while giving a fast
        # symbolic expression.
        V_fine = np.linspace(volt_grid.min(), volt_grid.max(), 2000)
        x_fine = _msmr_x_from_V(V_fine, params, msmr_T)

        # Sort by x for monotonic interpolation
        order = np.argsort(x_fine)
        x_fine = x_fine[order]
        V_fine = V_fine[order]

        # Remove duplicates
        unique = np.diff(x_fine, prepend=-np.inf) > 0
        x_fine = x_fine[unique]
        V_fine = V_fine[unique]

        # Store fitted params on the function for inspection
        def ocp_func(sto):
            return pb.Interpolant(
                x_fine,
                V_fine,
                sto,
                interpolator="pchip",
                extrapolate=False,
            )

        ocp_func.msmr_params = params.reshape(-1, 3)
        ocp_func.msmr_param_names = ["U0", "kappa", "Xj"]
        ocp_func.msmr_T = msmr_T

        # Also compute and store fit quality
        x_pred = _msmr_x_from_V(volt_grid, params, msmr_T)
        ocp_func.msmr_rmse_x = float(np.sqrt(np.mean((x_pred - sto_grid) ** 2)))

    elif ocp_mode == "interpolant":

        def ocp_func(sto):
            return pb.Interpolant(
                sto_grid,
                volt_grid,
                sto,
                interpolator="pchip",
                extrapolate=False,
            )

    else:
        raise ValueError(
            f"Unknown ocp_mode={ocp_mode!r}. "
            f"Choose from 'polynomial', 'msmr', or 'interpolant'."
        )

    return ocp_func


def _extract_sto_bounds_from_materials(
    materials: list[ActiveMaterial],
) -> tuple[float, float]:
    """Extract min/max stoichiometry bounds from active material OCP data (union for blends)."""
    bounds = []
    for mat in materials:
        if mat.has_ocv_data() and mat.soc_pct is not None:
            soc_pct_arr = np.array(mat.soc_pct)
            sto_min = soc_pct_arr.min() / 100.0
            sto_max = soc_pct_arr.max() / 100.0
            bounds.append((sto_min, sto_max))

    if not bounds:
        return 0.0, 1.0
    return min(b[0] for b in bounds), max(b[1] for b in bounds)


def _compute_stoichiometry_windows(
    cell_params: CellParametersInput,
    sim_params: SimulationParameters,
) -> StoichiometryWindows:
    """Compute electrode stoichiometry windows from equilibrium and coulombic loss."""
    ce_loss_pct = sim_params.first_cycle_coulombic_loss_pct

    # Extract stoichiometry bounds from OCP data
    sto_ca_min, sto_ca_max = _extract_sto_bounds_from_materials(
        cell_params.positive_electrode_active_materials
    )
    sto_an_min, sto_an_max = _extract_sto_bounds_from_materials(
        cell_params.negative_electrode_active_materials
    )

    # Calculate electrode-specific capacity: sum(capacity_i * mass_fraction_i)
    # This matches make_cell_design's electrode_specific_capacity calculation
    # where mass_fraction_i is the fraction of active material i within the electrode
    pos_spec_cap_weighted = sum(
        am.mass_fraction * am.specific_capacity_mAh_g
        for am in cell_params.positive_electrode_active_materials
    )
    neg_spec_cap_weighted = sum(
        am.mass_fraction * am.specific_capacity_mAh_g
        for am in cell_params.negative_electrode_active_materials
    )
    logger.info(
        "Electrode spec cap: pos_weighted=%.3f mAh/g (from %d materials), neg_weighted=%.3f mAh/g (from %d materials)",
        pos_spec_cap_weighted,
        len(cell_params.positive_electrode_active_materials),
        neg_spec_cap_weighted,
        len(cell_params.negative_electrode_active_materials),
    )
    for am in cell_params.positive_electrode_active_materials:
        logger.info(
            "  Pos AM: %s, mass_frac=%.4f, spec_cap=%.1f mAh/g, contribution=%.3f mAh/g",
            am.name,
            am.mass_fraction,
            am.specific_capacity_mAh_g,
            am.mass_fraction * am.specific_capacity_mAh_g,
        )

    positive_electrode_area_cm2 = _electrode_area_cm2(cell_params, is_positive=True)
    negative_electrode_area_cm2 = _electrode_area_cm2(cell_params, is_positive=False)
    # Use electrode-specific capacity directly (already weighted by mass fractions)
    # Formula: coating_sides * jelly_roll_count * sheet_count * area * mass_loading * spec_cap_weighted
    logger.info(
        "Capacity calc: pos area=%.2f cm², coating_sides=%d, jelly_roll=%d, sheets=%.1f, "
        "mass_loading=%.3f mg/cm², spec_cap_weighted=%.3f mAh/g",
        positive_electrode_area_cm2,
        cell_params.electrode_coating_side_count,
        cell_params.jelly_roll_count,
        cell_params.positive_electrode_sheet_count,
        cell_params.positive_electrode_mass_loading_mg_cm2,
        pos_spec_cap_weighted,
    )
    pos_cap_Ah = (
        cell_params.electrode_coating_side_count
        * cell_params.jelly_roll_count
        * cell_params.positive_electrode_sheet_count
        * positive_electrode_area_cm2
        * (cell_params.positive_electrode_mass_loading_mg_cm2 * 1e-3)  # mg/cm² to g/cm²
        * (pos_spec_cap_weighted * 1e-3)  # mAh/g to Ah/g
    )
    logger.info("Capacity calc: pos_cap_Ah=%.4f Ah", pos_cap_Ah)
    logger.info(
        "Capacity calc: neg area=%.2f cm², coating_sides=%d, jelly_roll=%d, sheets=%.1f, "
        "mass_loading=%.3f mg/cm², spec_cap_weighted=%.3f mAh/g",
        negative_electrode_area_cm2,
        cell_params.electrode_coating_side_count,
        cell_params.jelly_roll_count,
        cell_params.negative_electrode_sheet_count,
        cell_params.negative_electrode_mass_loading_mg_cm2,
        neg_spec_cap_weighted,
    )
    neg_cap_Ah = (
        cell_params.electrode_coating_side_count
        * cell_params.jelly_roll_count
        * cell_params.negative_electrode_sheet_count
        * negative_electrode_area_cm2
        * (cell_params.negative_electrode_mass_loading_mg_cm2 * 1e-3)  # mg/cm² to g/cm²
        * (neg_spec_cap_weighted * 1e-3)  # mAh/g to Ah/g
    )
    logger.info("Capacity calc: neg_cap_Ah=%.4f Ah", neg_cap_Ah)

    # Capacity ratios for stoichiometry scaling
    # Use min(pos_cap_Ah, neg_cap_Ah) directly to avoid rounding mismatches
    cell_capacity_Ah = min(pos_cap_Ah, neg_cap_Ah)
    ca_ratio = cell_capacity_Ah / pos_cap_Ah if pos_cap_Ah > 0 else 0.0
    an_ratio = cell_capacity_Ah / neg_cap_Ah if neg_cap_Ah > 0 else 0.0

    # Stoichiometry window ranges (from OCP data)
    sto_ca_range = sto_ca_max - sto_ca_min
    sto_an_range = sto_an_max - sto_an_min

    # Stoichiometry deltas accounting for coulombic loss and cell vs electrode capacity
    # dsto_cell: fraction of stoichiometry window used by cell capacity (scaled by efficiency)
    # dsto_loss: fraction lost to coulombic efficiency
    efficiency = 1.0 - (ce_loss_pct / 100.0)

    dsto_cell_ca = ca_ratio * sto_ca_range * efficiency
    dsto_loss_ca = (ce_loss_pct / 100.0) * ca_ratio * sto_ca_range

    dsto_cell_an = an_ratio * sto_an_range * efficiency
    dsto_loss_an = (ce_loss_pct / 100.0) * an_ratio * sto_an_range

    # Stoichiometry at 0% (discharged) and 100% SOC (charged)
    # Cathode: sto decreases with lithiation (high sto = high V / delithiated)
    sto_ca0 = sto_ca_max - dsto_loss_ca
    sto_ca100 = sto_ca0 - dsto_cell_ca

    # Anode: sto increases with lithiation (high sto = high lithiation / low V)
    sto_an0 = sto_an_min + dsto_loss_an
    sto_an100 = sto_an0 + dsto_cell_an

    return StoichiometryWindows(
        sto_ca0=sto_ca0,
        sto_ca100=sto_ca100,
        sto_an0=sto_an0,
        sto_an100=sto_an100,
    )


def _init_sto_overrides_from_soc(
    param: pybamm.ParameterValues,
    sto_windows: StoichiometryWindows,
    target_soc: float,
    working_electrode: str | None = None,
) -> dict[str, float]:
    """Compute Initial concentration param overrides from stoichiometry windows.

    Maps target SOC [0, 1] to electrode stoichiometry using our equilibrium
    windows, then to mol/m³ via max concentration. Use these overrides (do NOT
    pass initial_soc to solve) so PyBaMM starts from our init sto.

    For half-cell mode (working_electrode='positive') only the
    working electrode's concentration is set; the counter electrode (lithium metal)
    has no stoichiometry window.
    """
    sto_ca = sto_windows.sto_ca0 + target_soc * (
        sto_windows.sto_ca100 - sto_windows.sto_ca0
    )
    max_conc_pos = param["Maximum concentration in positive electrode [mol.m-3]"]
    overrides: dict[str, float] = {
        "Initial concentration in positive electrode [mol.m-3]": sto_ca * max_conc_pos,
    }
    if working_electrode != "positive":
        # In a positive half-cell the negative electrode is lithium metal (counter);
        # PyBaMM does not accept a stoichiometry override for it.
        sto_an = sto_windows.sto_an0 + target_soc * (
            sto_windows.sto_an100 - sto_windows.sto_an0
        )
        max_conc_neg = param["Maximum concentration in negative electrode [mol.m-3]"]
        overrides["Initial concentration in negative electrode [mol.m-3]"] = (
            sto_an * max_conc_neg
        )
    return overrides


def _apply_soc_to_param(
    param: pybamm.ParameterValues,
    sto_windows: StoichiometryWindows | None,
    soc: float,
    working_electrode: str | None,
) -> None:
    """Apply stoichiometry concentration overrides to *param* in-place for *soc*.

    No-op when sto_windows is None (PyBaMM will use initial_soc instead).
    """
    if sto_windows is not None:
        param.update(
            _init_sto_overrides_from_soc(param, sto_windows, soc, working_electrode),
            check_already_exists=False,
        )


def _solve_sim(
    sim: Any,
    solver: Any,
    sto_windows: StoichiometryWindows | None,
    initial_soc: float,
) -> Any:
    """Solve *sim*; omit initial_soc when sto_windows are set.

    When sto_windows are set, concentration overrides in the parameter values
    already encode the target SOC — passing initial_soc would cause PyBaMM to
    override them.
    """
    if sto_windows is not None:
        return sim.solve(solver=solver)
    return sim.solve(initial_soc=initial_soc, solver=solver)


def _compute_equilibrium_kpis(
    cell_params: CellParametersInput,
    sim_params: SimulationParameters | None = None,
) -> CellEquilibriumKPIs:
    """Compute equilibrium KPIs."""
    positive_electrode_area_cm2 = _electrode_area_cm2(cell_params, is_positive=True)
    negative_electrode_area_cm2 = _electrode_area_cm2(cell_params, is_positive=False)
    cell_volume_L = _cell_volume_L(cell_params)

    # FIX: capacity-weighted voltage for blended electrodes
    pos_voltage = _capacity_weighted_voltage_V(
        cell_params.positive_electrode_active_materials
    )
    neg_voltage = _capacity_weighted_voltage_V(
        cell_params.negative_electrode_active_materials
    )
    cell_voltage_V = pos_voltage - neg_voltage

    # Calculate electrode-specific capacity: sum(capacity_i * mass_fraction_i)
    # This matches make_cell_design's electrode_specific_capacity calculation
    # where mass_fraction_i is the fraction of active material i within the electrode
    pos_spec_cap_weighted = sum(
        am.mass_fraction * am.specific_capacity_mAh_g
        for am in cell_params.positive_electrode_active_materials
    )
    neg_spec_cap_weighted = sum(
        am.mass_fraction * am.specific_capacity_mAh_g
        for am in cell_params.negative_electrode_active_materials
    )

    pos_coating_density = (
        cell_params.positive_electrode_mass_loading_mg_cm2
        * 10.0
        / cell_params.positive_coating_thickness_um
        if cell_params.positive_coating_thickness_um > 0
        else 0.0
    )
    neg_coating_density = (
        cell_params.negative_electrode_mass_loading_mg_cm2
        * 10.0
        / cell_params.negative_coating_thickness_um
        if cell_params.negative_coating_thickness_um > 0
        else 0.0
    )
    pos_formulation_density = _formulation_density_g_cm3(
        cell_params.positive_electrode_active_materials,
        cell_params.positive_electrode_binders,
        cell_params.positive_electrode_conductive_agents,
    )
    neg_formulation_density = _formulation_density_g_cm3(
        cell_params.negative_electrode_active_materials,
        cell_params.negative_electrode_binders,
        cell_params.negative_electrode_conductive_agents,
    )

    logger.info(
        "Electrode calc: pos mass_loading=%.3f mg/cm2 thickness=%.1f um -> "
        "coating_density=%.3f g/cm3; formulation_density=%.3f g/cm3",
        cell_params.positive_electrode_mass_loading_mg_cm2,
        cell_params.positive_coating_thickness_um,
        pos_coating_density,
        pos_formulation_density,
    )
    logger.info(
        "Electrode calc: neg mass_loading=%.3f mg/cm2 thickness=%.1f um -> "
        "coating_density=%.3f g/cm3; formulation_density=%.3f g/cm3",
        cell_params.negative_electrode_mass_loading_mg_cm2,
        cell_params.negative_coating_thickness_um,
        neg_coating_density,
        neg_formulation_density,
    )

    # Use electrode-specific capacity directly (already weighted by mass fractions)
    # Formula: coating_sides * jelly_roll_count * sheet_count * area * mass_loading * spec_cap_weighted
    pos_cap_Ah = (
        cell_params.electrode_coating_side_count
        * cell_params.jelly_roll_count
        * cell_params.positive_electrode_sheet_count
        * positive_electrode_area_cm2
        * (cell_params.positive_electrode_mass_loading_mg_cm2 * 1e-3)  # mg/cm² to g/cm²
        * (pos_spec_cap_weighted * 1e-3)  # mAh/g to Ah/g
    )
    neg_cap_Ah = (
        cell_params.electrode_coating_side_count
        * cell_params.jelly_roll_count
        * cell_params.negative_electrode_sheet_count
        * negative_electrode_area_cm2
        * (cell_params.negative_electrode_mass_loading_mg_cm2 * 1e-3)  # mg/cm² to g/cm²
        * (neg_spec_cap_weighted * 1e-3)  # mAh/g to Ah/g
    )

    capacity_Ah = min(pos_cap_Ah, neg_cap_Ah)
    energy_Wh = capacity_Ah * cell_voltage_V

    # Porosity (clamped to [0, 1] for physical validity)
    pos_porosity_raw = (
        1 - (pos_coating_density / pos_formulation_density)
        if pos_formulation_density > 0
        else 0.0
    )
    neg_porosity_raw = (
        1 - (neg_coating_density / neg_formulation_density)
        if neg_formulation_density > 0
        else 0.0
    )
    pos_porosity = max(0.0, min(1.0, pos_porosity_raw))
    neg_porosity = max(0.0, min(1.0, neg_porosity_raw))

    logger.info(
        "Porosity calc: pos 1 - (coating/formulation) = 1 - (%.4f/%.4f) -> raw=%.6f clamped=%.6f",
        pos_coating_density,
        pos_formulation_density,
        pos_porosity_raw,
        pos_porosity,
    )
    logger.info(
        "Porosity calc: neg 1 - (coating/formulation) = 1 - (%.4f/%.4f) -> raw=%.6f clamped=%.6f",
        neg_coating_density,
        neg_formulation_density,
        neg_porosity_raw,
        neg_porosity,
    )

    # Electrode volume fractions (true volume fractions)
    pos_am_vf, pos_binder_vf, pos_cond_vf = _electrode_volume_fractions(
        pos_porosity,
        cell_params.positive_electrode_active_materials,
        cell_params.positive_electrode_binders,
        cell_params.positive_electrode_conductive_agents,
    )
    neg_am_vf, neg_binder_vf, neg_cond_vf = _electrode_volume_fractions(
        neg_porosity,
        cell_params.negative_electrode_active_materials,
        cell_params.negative_electrode_binders,
        cell_params.negative_electrode_conductive_agents,
    )

    # Sanity: porosity + volume fractions = 1
    for label, porosity, am_vf, binder_vf, cond_vf in [
        (
            "Positive",
            pos_porosity,
            pos_am_vf,
            pos_binder_vf,
            pos_cond_vf,
        ),
        (
            "Negative",
            neg_porosity,
            neg_am_vf,
            neg_binder_vf,
            neg_cond_vf,
        ),
    ]:
        total = porosity + am_vf + binder_vf + cond_vf
        if abs(total - 1.0) >= 1e-6:
            raise ValueError(
                f"{label} electrode: porosity+vol_fracs must sum to 1, got {total:.10f}"
            )

    # Mass model

    pos_coat = _coat_mass(
        cell_params.electrode_coating_side_count,
        cell_params.jelly_roll_count,
        cell_params.positive_electrode_sheet_count,
        positive_electrode_area_cm2,
        cell_params.positive_electrode_mass_loading_mg_cm2,
    )
    neg_coat = _coat_mass(
        cell_params.electrode_coating_side_count,
        cell_params.jelly_roll_count,
        cell_params.negative_electrode_sheet_count,
        negative_electrode_area_cm2,
        cell_params.negative_electrode_mass_loading_mg_cm2,
    )
    pos_foil = _foil_mass(
        positive_electrode_area_cm2,
        cell_params.positive_electrode_foil_thickness_um,
        cell_params.positive_electrode_foil_density_g_cm3,
        cell_params.jelly_roll_count,
        cell_params.positive_electrode_sheet_count,
    )
    neg_foil = _foil_mass(
        negative_electrode_area_cm2,
        cell_params.negative_electrode_foil_thickness_um,
        cell_params.negative_electrode_foil_density_g_cm3,
        cell_params.jelly_roll_count,
        cell_params.negative_electrode_sheet_count,
    )

    sep_vol_cm3 = (
        cell_params.separator_sheet_count
        * cell_params.separator_sheet_width_mm
        * cell_params.separator_sheet_height_mm
        / 100.0
        * (cell_params.separator_thickness_um * 1e-4)
    )
    sep_mass = (
        cell_params.jelly_roll_count
        * sep_vol_cm3
        * cell_params.separator_density_g_cm3
        * (1 - cell_params.separator_porosity)
    )
    jelly_roll_mass_g = pos_coat + neg_coat + pos_foil + neg_foil + sep_mass

    pos_coat_vol = _coat_vol(
        cell_params.electrode_coating_side_count,
        cell_params.jelly_roll_count,
        cell_params.positive_electrode_sheet_count,
        positive_electrode_area_cm2,
        cell_params.positive_coating_thickness_um,
    )
    neg_coat_vol = _coat_vol(
        cell_params.electrode_coating_side_count,
        cell_params.jelly_roll_count,
        cell_params.negative_electrode_sheet_count,
        negative_electrode_area_cm2,
        cell_params.negative_coating_thickness_um,
    )
    pore_vol = (
        pos_porosity * pos_coat_vol
        + neg_porosity * neg_coat_vol
        + cell_params.separator_porosity * sep_vol_cm3
    )
    if cell_params.electrolyte is None:
        electrolyte_density = cell_params.electrolyte_density_g_cm3
    else:
        comp = cell_params.electrolyte.composition
        if comp.solvents:
            inv_rho = sum(s.vol_frac / s.density_g_cm3 for s in comp.solvents)
            electrolyte_density = (
                1.0 / inv_rho if inv_rho > 0 else cell_params.electrolyte_density_g_cm3
            )
        elif comp.salts:
            electrolyte_density = comp.salts[0].density_g_cm3
        else:
            electrolyte_density = cell_params.electrolyte_density_g_cm3
    electrolyte_mass_g = (
        pore_vol * cell_params.electrolyte_fill_ratio * electrolyte_density
    )

    # Casing
    if (
        cell_params.form_factor.lower() == "cylindrical"
        and cell_params.cell_diameter_mm
        and cell_params.cell_height_mm
    ):
        d, h = cell_params.cell_diameter_mm, cell_params.cell_height_mm
        surf_mm2 = pi * d * h + 2 * pi * (d / 2) ** 2
    elif (
        cell_params.cell_width_mm
        and cell_params.cell_height_mm
        and cell_params.cell_thickness_mm
    ):
        w, h, t = (
            cell_params.cell_width_mm,
            cell_params.cell_height_mm,
            cell_params.cell_thickness_mm,
        )
        surf_mm2 = 2 * (w * h + w * t + h * t)
    else:
        surf_mm2 = 0.0
    casing_mass_g = (
        surf_mm2 * cell_params.casing_thickness_mm / 1000
    ) * cell_params.casing_density_g_cm3

    cell_mass_g = jelly_roll_mass_g + electrolyte_mass_g + casing_mass_g
    if cell_mass_g <= 0:
        cell_mass_g = 1.0

    cell_n_p_ratio = neg_cap_Ah / pos_cap_Ah if pos_cap_Ah > 0 else 0.0

    # Electrode volume fractions in cell: (coating + foil volume) / cell volume
    cell_volume_cm3 = cell_volume_L * 1000.0
    pos_foil_vol_cm3 = (
        pos_foil / cell_params.positive_electrode_foil_density_g_cm3
        if cell_params.positive_electrode_foil_density_g_cm3 > 0
        else 0.0
    )
    neg_foil_vol_cm3 = (
        neg_foil / cell_params.negative_electrode_foil_density_g_cm3
        if cell_params.negative_electrode_foil_density_g_cm3 > 0
        else 0.0
    )
    pos_electrode_vol_cm3 = pos_coat_vol + pos_foil_vol_cm3
    neg_electrode_vol_cm3 = neg_coat_vol + neg_foil_vol_cm3
    pos_vol_frac_in_cell = (
        pos_electrode_vol_cm3 / cell_volume_cm3 if cell_volume_cm3 > 0 else 0.0
    )
    neg_vol_frac_in_cell = (
        neg_electrode_vol_cm3 / cell_volume_cm3 if cell_volume_cm3 > 0 else 0.0
    )
    pos_vol_frac_in_cell = max(0.0, min(1.0, pos_vol_frac_in_cell))
    neg_vol_frac_in_cell = max(0.0, min(1.0, neg_vol_frac_in_cell))

    cell_mass_kg = cell_mass_g / 1000.0
    grav_Wh_kg = energy_Wh / cell_mass_kg if cell_mass_kg > 0 else 0.0
    vol_Wh_L = energy_Wh / cell_volume_L if cell_volume_L > 0 else 0.0
    sto_windows = _compute_stoichiometry_windows(cell_params, sim_params=sim_params)

    # Compute bill of materials
    bom = _compute_bill_of_materials(
        cell_params=cell_params,
        cell_mass_g=cell_mass_g,
        electrolyte_mass_g=electrolyte_mass_g,
        casing_mass_g=casing_mass_g,
    )

    out = CellEquilibriumKPIs(
        cell_theoretical_capacity_Ah=round(capacity_Ah, 4),
        cell_theoretical_energy_Wh=round(energy_Wh, 4),
        cell_n_p_ratio=round(cell_n_p_ratio, 4),
        stoichiometry_windows=sto_windows,
        cell_mass_g=round(cell_mass_g, 4),
        cell_volume_L=round(cell_volume_L, 4),
        cell_theoretical_gravimetric_energy_density_Wh_kg=round(grav_Wh_kg, 4),
        cell_theoretical_volumetric_energy_density_Wh_L=round(vol_Wh_L, 4),
        positive_electrode_porosity=round(pos_porosity, 4),
        negative_electrode_porosity=round(neg_porosity, 4),
        positive_electrode_density_g_cm3=round(pos_coating_density, 4),
        negative_electrode_density_g_cm3=round(neg_coating_density, 4),
        positive_electrode_volume_fraction=round(pos_vol_frac_in_cell, 4),
        negative_electrode_volume_fraction=round(neg_vol_frac_in_cell, 4),
        electrolyte_mass_g=round(electrolyte_mass_g, 4),
        electrolyte_mass_fraction=round(
            electrolyte_mass_g / cell_mass_g if cell_mass_g > 0 else 0.0, 4
        ),
        jelly_roll_mass_g=round(jelly_roll_mass_g, 4),
        jelly_roll_mass_fraction=round(
            jelly_roll_mass_g / cell_mass_g if cell_mass_g > 0 else 0.0, 4
        ),
        casing_mass_g=round(casing_mass_g, 4),
        casing_mass_fraction=round(
            casing_mass_g / cell_mass_g if cell_mass_g > 0 else 0.0, 4
        ),
        bill_of_materials=bom,
    )
    return out


def _build_ocp_interpolant(
    mat: ActiveMaterial,
    pb: Any,
    *,
    is_cathode: bool = False,
    ocp_mode: str = "polynomial",
    poly_deg: int | None = None,
    msmr_n_sites: int | None = None,
    msmr_T: float = 298.15,
) -> Any:
    """Build an OCP callable for a single active material.

    Args:
        mat: Active material with OCV data
        pb: PyBaMM module
        is_cathode: Whether this is a cathode material
        ocp_mode: OCP model type ("polynomial", "msmr", or "interpolant")
        poly_deg: Polynomial degree for polynomial mode. If None and ocp_mode is "polynomial",
                  automatically selects based on material name.
        msmr_n_sites: Number of sites for MSMR model
        msmr_T: Temperature for MSMR model [K]

    Returns:
        PyBaMM callable function for OCP
    """
    # Resolve poly_deg for polynomial mode if not provided
    if ocp_mode == "polynomial" and poly_deg is None:
        # Use material name to determine appropriate polynomial degree
        poly_deg = _poly_deg_for_chemistry(mat.name)

    soc = np.array(mat.soc_pct) / 100.0
    volt = np.array(mat.dch_ocv_V)

    if is_cathode:
        soc = 1.0 - soc

    order = np.argsort(soc)
    soc = soc[order]
    volt = volt[order]

    # Remove duplicate stoichiometry values
    unique_mask = np.diff(soc, prepend=-np.inf) > 0
    soc = soc[unique_mask]
    volt = volt[unique_mask]

    return _make_ocp_func(
        soc,
        volt,
        pb,
        ocp_mode=ocp_mode,
        poly_deg=poly_deg,
        msmr_n_sites=msmr_n_sites,
        msmr_T=msmr_T,
    )


def _build_blend_ocp_interpolant(
    materials: list[ActiveMaterial],
    pb: Any,
    *,
    is_cathode: bool = False,
    n_voltage_steps: int = 1000,  # Increased steps for better derivative resolution
    ocp_mode: str = "polynomial",
    poly_deg: int | None = None,
    msmr_n_sites: int | None = None,
    msmr_T: float = 298.15,
) -> Any | None:
    """Build composite OCP via volumetric integration of dX/dOCV.

    Requires max_lithium_conc_mol_m3 to be set for all materials with OCV data,
    as it's needed for volumetric capacity scaling in blend calculations.

    Args:
        materials: List of active materials to blend
        pb: PyBaMM module
        is_cathode: Whether these are cathode materials
        n_voltage_steps: Number of voltage steps for integration
        ocp_mode: OCP model type ("polynomial", "msmr", or "interpolant")
        poly_deg: Polynomial degree for polynomial mode. If None and ocp_mode is "polynomial",
                  automatically selects based on material names (uses maximum degree).
        msmr_n_sites: Number of sites for MSMR model
        msmr_T: Temperature for MSMR model [K]

    Returns:
        PyBaMM callable function for blended OCP, or None if no valid materials
    """

    # Check for materials with OCV data but missing max_lithium_conc_mol_m3
    ocv_mats_missing_conc = [
        m for m in materials if m.has_ocv_data() and m.max_lithium_conc_mol_m3 is None
    ]
    if ocv_mats_missing_conc:
        missing_names = [m.name for m in ocv_mats_missing_conc]
        raise ValueError(
            f"Blend OCP construction requires max_lithium_conc_mol_m3 for all materials "
            f"with OCV data. Missing for: {', '.join(missing_names)}. "
            f"Please provide max_lithium_conc_mol_m3 [mol/m³] for these materials."
        )

    # Filter materials: must have OCV data AND max_lithium_conc_mol_m3 (required for blend scaling)
    ocv_mats = [
        m
        for m in materials
        if m.has_ocv_data() and m.max_lithium_conc_mol_m3 is not None
    ]
    if not ocv_mats:
        return None

    # Resolve poly_deg for polynomial mode if not provided
    if ocp_mode == "polynomial" and poly_deg is None:
        # Use material names to determine appropriate polynomial degrees, take maximum
        degrees = [_poly_deg_for_chemistry(m.name) for m in ocv_mats]
        poly_deg = max(degrees)

    # 1. Determine Global Voltage Window
    v_mins = [np.min(m.dch_ocv_V) for m in ocv_mats]
    v_maxs = [np.max(m.dch_ocv_V) for m in ocv_mats]
    V_min, V_max = min(v_mins), max(v_maxs)

    V_grid = np.linspace(V_min, V_max, n_voltage_steps)

    # 2. Compute the Weighted Differential Sum: Σ (dX/dV * Cmax * mf / density)
    # This represents the total differential capacity of the blend (Ah/L-electrode)
    dQ_dV_blend = np.zeros_like(V_grid)

    for m in ocv_mats:
        # Sort raw data to ensure monotonicity for interpolation
        sort_idx = np.argsort(m.dch_ocv_V)
        v_raw = np.array(m.dch_ocv_V)[sort_idx]
        x_raw = np.array(m.soc_pct)[sort_idx] / 100.0  # Stoichiometry X (0 to 1)

        # Interpolate X onto the common V_grid
        interp_x = interp1d(
            v_raw, x_raw, bounds_error=False, fill_value=(x_raw[0], x_raw[-1])
        )(V_grid)

        # Calculate dX/dOCV
        dx_dv = np.gradient(interp_x, V_grid)

        # Apply your formula scaling: (Cmax_i * mass_frac_i) / density_i
        # Requires max_lithium_conc_mol_m3 to be set
        scaling = (
            (m.max_lithium_conc_mol_m3 * m.mass_fraction) / m.density_g_cm3 * _F / 3600
        )

        dQ_dV_blend += dx_dv * scaling

    # 3. Integrate to get Q_blend (Volumetric Capacity)
    # Using cumulative trapezoidal integration

    Q_blend = cumtrapz(dQ_dV_blend, V_grid, initial=0)

    total_Q = np.abs(Q_blend[-1] - Q_blend[0])
    if total_Q <= 0:
        return None

    # 4. Map to Normalized Stoichiometry (0 to 1) for PyBaMM
    # This represents the aggregate state of the blended electrode
    sto_grid = (Q_blend - np.min(Q_blend)) / total_Q

    # 5. Handle directionality (Voltage vs Stoichiometry)
    # Ensure sto_grid increases monotonically while V_grid is mapped correctly
    if is_cathode:
        # For cathodes, V usually increases with SOC (lithiation decreases V)
        # sto_grid already increases with V_grid, so natural order is correct
        blend_volt = V_grid
    else:
        # For anodes, V decreases with SOC but sto increases with SOC
        # Reverse voltage to match increasing stoichiometry, then sort to ensure monotonicity
        blend_volt = V_grid[::-1]
        sto_grid = sto_grid[::-1]
        # Sort by sto_grid to ensure monotonicity (required for interpolation stability)
        sort_idx = np.argsort(sto_grid)
        sto_grid = sto_grid[sort_idx]
        blend_volt = blend_volt[sort_idx]

    # Remove duplicates for interpolation stability
    unique_mask = np.diff(sto_grid, prepend=-np.inf) > 1e-8
    sto_grid = sto_grid[unique_mask]
    blend_volt = blend_volt[unique_mask]

    return _make_ocp_func(
        sto_grid,
        blend_volt,
        pb,
        ocp_mode=ocp_mode,
        poly_deg=poly_deg,
        msmr_n_sites=msmr_n_sites,
        msmr_T=msmr_T,
    )


def _inject_reference_electrode_params(
    param: Any,
    user_j0: float | None,
    user_ocp_V: float | None,
    user_alpha: float | None,
) -> None:
    """Inject counter/reference electrode parameters required for half-cell mode.

    Full-cell parameter sets (e.g. Chen2020, Prada2013) do not include these keys.
    We pull defaults from the half-cell SPMe model's own default_parameter_values
    (which assume a lithium metal counter electrode) and allow the user to override
    exchange-current density, OCP, and charge transfer coefficient with constant
    scalars to model any reference electrode (Li metal, Na metal, pseudo-reference, etc.).
    """
    half_cell_defaults = _get_halfcell_spme_defaults()

    # Always inject the partial molar volume (no user override needed).
    param.update(
        {
            "Lithium metal partial molar volume [m3.mol-1]": half_cell_defaults[
                "Lithium metal partial molar volume [m3.mol-1]"
            ],
        },
        check_already_exists=False,
    )

    # Exchange-current density: use Xu2019 function or user constant.
    # Inject under both known PyBaMM key names for cross-version compatibility.
    j0_value = (
        float(user_j0)
        if user_j0 is not None
        else half_cell_defaults[
            "Exchange-current density for lithium metal electrode [A.m-2]"
        ]
    )
    param.update(
        {
            "Exchange-current density for lithium metal electrode [A.m-2]": j0_value,
            "Lithium metal electrode exchange current density [A.m-2]": j0_value,
        },
        check_already_exists=False,
    )

    # OCP of the counter electrode: 0.0 V for pure lithium metal (default).
    param.update(
        {
            "Negative electrode OCP [V]": float(user_ocp_V)
            if user_ocp_V is not None
            else 0.0
        },
        check_already_exists=False,
    )

    # Butler-Volmer charge transfer coefficient: always inject the documented default
    # (0.5 = symmetric kinetics) so half-cell mode is not silently influenced by
    # whatever the full-cell parameter set had for the negative electrode.
    param.update(
        {
            "Negative electrode charge transfer coefficient": float(user_alpha)
            if user_alpha is not None
            else 0.5
        },
        check_already_exists=False,
    )


def _build_pybamm_params_from_equilibrium(
    cell_params: CellParametersInput,
    cell_equilibrium_kpis: CellEquilibriumKPIs,
    sim_params: SimulationParameters,
) -> pybamm.ParameterValues:
    pb = pybamm
    # Check individual material names for LFP detection (not joined string)
    # This prevents false positives from materials like "NLFP-blend" or "Graphite+LFP"
    has_lfp = any(
        "LFP" in am.name.upper()
        for am in cell_params.positive_electrode_active_materials
    )

    # For LFP chemistries, always use Prada2013 as the base parameter set (even when
    # use_pybamm_params is "custom"), otherwise:
    # - when use_pybamm_params is "custom", load Chen2020 as base and override with user values
    # - when use_pybamm_params is not "custom", use the specified parameter set
    if has_lfp:
        # If the user explicitly requested a different set, warn and override to Prada2013.
        if sim_params.use_pybamm_params and sim_params.use_pybamm_params != "Prada2013":
            logger.warning(
                "Auto-selecting Prada2013 parameter set for LFP chemistry "
                "(user requested '%s')",
                sim_params.use_pybamm_params,
            )
            param_set = "Prada2013"
        # When empty / falsy, we're using Prada2013 as the base and overriding with
        # user-provided material properties.
        if not sim_params.use_pybamm_params or sim_params.use_pybamm_params == "custom":
            logger.warning(
                "Detected LFP chemistry; loading Prada2013 parameter set as base; "
                "will override with user-provided material properties"
            )
        param_set = "Prada2013"
    elif not sim_params.use_pybamm_params or sim_params.use_pybamm_params == "custom":
        param_set = "Chen2020"
        logger.warning(
            "Loading Chen2020 parameter set as base; will override with user-provided material properties"
        )
    else:
        param_set = sim_params.use_pybamm_params

    param = pb.ParameterValues(param_set)

    electrode_area_cm2 = _electrode_area_cm2(cell_params, is_positive=True)

    area_m2 = electrode_area_cm2 / 10000.0
    side = max(1e-4, sqrt(area_m2))
    n_elec = (
        cell_params.electrode_coating_side_count
        * cell_params.jelly_roll_count
        * cell_params.positive_electrode_sheet_count
    )
    # Electrode volume fractions (true volume fractions)
    pos_am_vf, pos_binder_vf, pos_cond_vf = _electrode_volume_fractions(
        cell_equilibrium_kpis.positive_electrode_porosity,
        cell_params.positive_electrode_active_materials,
        cell_params.positive_electrode_binders,
        cell_params.positive_electrode_conductive_agents,
    )
    neg_am_vf, neg_binder_vf, neg_cond_vf = _electrode_volume_fractions(
        cell_equilibrium_kpis.negative_electrode_porosity,
        cell_params.negative_electrode_active_materials,
        cell_params.negative_electrode_binders,
        cell_params.negative_electrode_conductive_agents,
    )

    electrode_params = {
        "Nominal cell capacity [A.h]": cell_equilibrium_kpis.cell_theoretical_capacity_Ah,
        "Number of electrodes connected in parallel to make a cell": n_elec,
        "Electrode width [m]": side,
        "Electrode height [m]": side,
        "Electrode length [m]": 1.0,
        "Positive electrode thickness [m]": cell_params.positive_coating_thickness_um
        / 1e6,
        "Positive electrode porosity": cell_equilibrium_kpis.positive_electrode_porosity,
        "Positive electrode active material volume fraction": pos_am_vf,
        "Positive electrode density [kg.m-3]": cell_equilibrium_kpis.positive_electrode_density_g_cm3
        * 1000,
        "Positive current collector thickness [m]": cell_params.positive_electrode_foil_thickness_um
        / 1e6,
        "Positive current collector density [kg.m-3]": cell_params.positive_electrode_foil_density_g_cm3
        * 1000,
        "Negative electrode thickness [m]": cell_params.negative_coating_thickness_um
        / 1e6,
        "Negative electrode porosity": cell_equilibrium_kpis.negative_electrode_porosity,
        "Negative electrode active material volume fraction": neg_am_vf,
        "Negative electrode density [kg.m-3]": cell_equilibrium_kpis.negative_electrode_density_g_cm3
        * 1000,
        "Negative current collector thickness [m]": cell_params.negative_electrode_foil_thickness_um
        / 1e6,
        "Negative current collector density [kg.m-3]": cell_params.negative_electrode_foil_density_g_cm3
        * 1000,
        "Separator thickness [m]": cell_params.separator_thickness_um / 1e6,
        "Separator porosity": cell_params.separator_porosity,
        "Separator density [kg.m-3]": cell_params.separator_density_g_cm3 * 1000,
        "Upper voltage cut-off [V]": sim_params.upper_voltage_cutoff_V,
        "Lower voltage cut-off [V]": sim_params.lower_voltage_cutoff_V,
        "Contact resistance [Ohm]": (
            sim_params.cell_contact_resistance_Ohm
            if sim_params.cell_contact_resistance_Ohm is not None
            else (
                cell_params.cell_contact_resistance_Ohm
                if cell_params.cell_contact_resistance_Ohm is not None
                else 1e-4
            )
        ),
        "Cell cooling surface area [m2]": sim_params.cell_cooling_surface_area_m2,
        "Total heat transfer coefficient [W.m-2.K-1]": sim_params.cell_heat_transfer_coefficient_W_m2_K,
        "Ambient temperature [K]": sim_params.ambient_temperature_K,
        "Initial temperature [K]": (
            sim_params.initial_cell_temperature_K or sim_params.ambient_temperature_K
        ),
        "Reference temperature [K]": sim_params.reference_cell_temperature_K,
        # Current collector conductivity & specific heat — required by lumped
        # thermal model but missing from some parameter sets (e.g. Chen2020).
        # Defaults: Al (positive) and Cu (negative) at 25°C.
        "Positive current collector conductivity [S.m-1]": cell_params.positive_electrode_foil_electronic_conductivity_S_m,  # Al
        "Negative current collector conductivity [S.m-1]": cell_params.negative_electrode_foil_electronic_conductivity_S_m,  # Cu
        "Positive current collector specific heat capacity [J.kg-1.K-1]": cell_params.positive_electrode_foil_specific_heat_capacity_J_kg_K,  # Al
        "Negative current collector specific heat capacity [J.kg-1.K-1]": cell_params.negative_electrode_foil_specific_heat_capacity_J_kg_K,  # Cu
        "Positive electrode specific heat capacity [J.kg-1.K-1]": cell_params.positive_electrode_specific_heat_capacity_J_kg_K,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": cell_params.negative_electrode_specific_heat_capacity_J_kg_K,
        "Separator specific heat capacity [J.kg-1.K-1]": cell_params.separator_specific_heat_capacity_J_kg_K,
        # Electrode electronic conductivity (solid-phase) — needed by SPMe.
        # Defaults: NMC-class cathode and graphite anode.
        "Positive electrode conductivity [S.m-1]": cell_params.positive_electrode_electronic_conductivity_S_m,
        "Negative electrode conductivity [S.m-1]": cell_params.negative_electrode_electronic_conductivity_S_m,
    }

    vol_L = _cell_volume_L(cell_params)
    electrode_params["Cell volume [m3]"] = vol_L / 1000.0

    param.update(electrode_params, check_already_exists=False)

    # Half-cell mode: inject lithium metal counter-electrode parameters.
    # Chen2020 (and most full-cell sets) do not include these; they live in the
    # half-cell model's own default_parameter_values.
    if sim_params.working_electrode == "positive":
        _inject_reference_electrode_params(
            param,
            sim_params.reference_electrode_exchange_current_density_A_m2,
            sim_params.reference_electrode_ocp_V,
            sim_params.reference_electrode_charge_transfer_coefficient,
        )

    # Double-layer capacitance per-electrode overrides.
    if sim_params.enable_double_layer_capacitance:
        cdl_overrides: dict[str, float] = {}
        if sim_params.positive_electrode_double_layer_capacitance_F_m2 is not None:
            cdl_overrides["Positive electrode double-layer capacity [F.m-2]"] = (
                sim_params.positive_electrode_double_layer_capacitance_F_m2
            )
        if sim_params.negative_electrode_double_layer_capacitance_F_m2 is not None:
            cdl_overrides["Negative electrode double-layer capacity [F.m-2]"] = (
                sim_params.negative_electrode_double_layer_capacitance_F_m2
            )
        if cdl_overrides:
            param.update(cdl_overrides, check_already_exists=False)

    for overrides in [
        _electrolyte_param_overrides(cell_params),
        _active_material_param_overrides(cell_params),
        _tortuosity_param_overrides(cell_params),
    ]:
        if overrides:
            param.update(overrides, check_already_exists=False)

    # OCV overrides:
    # - Historically only ran when use_pybamm_params == "custom"
    # - Default/migration configs use "" / None, so treat those as "custom" too
    if sim_params.use_pybamm_params in (None, "", "custom"):
        # FIX: use blend OCP for blended electrodes
        pos_mats_with_ocv = [
            m
            for m in cell_params.positive_electrode_active_materials
            if m.has_ocv_data()
        ]
        neg_mats_with_ocv = [
            m
            for m in cell_params.negative_electrode_active_materials
            if m.has_ocv_data()
        ]

        # Build OCP functions for each electrode
        pos_ocp_func = None
        neg_ocp_func = None

        if len(pos_mats_with_ocv) > 1:
            blend_ocp = _build_blend_ocp_interpolant(
                cell_params.positive_electrode_active_materials,
                pb,
                is_cathode=True,
                ocp_mode=sim_params.positive_electrode_ocv_model,
            )
            if blend_ocp is not None:
                pos_ocp_func = blend_ocp
                param.update(
                    {"Positive electrode OCP [V]": blend_ocp},
                    check_already_exists=False,
                )
        elif len(pos_mats_with_ocv) == 1:
            pos_ocp_func = _build_ocp_interpolant(
                pos_mats_with_ocv[0],
                pb,
                is_cathode=True,
                ocp_mode=sim_params.positive_electrode_ocv_model,
            )
            param.update(
                {"Positive electrode OCP [V]": pos_ocp_func},
                check_already_exists=False,
            )
            if pos_mats_with_ocv[0].max_lithium_conc_mol_m3 is not None:
                param.update(
                    {
                        "Maximum concentration in positive electrode [mol.m-3]": pos_mats_with_ocv[
                            0
                        ].max_lithium_conc_mol_m3
                    },
                    check_already_exists=False,
                )

        # In half-cell mode (working_electrode='positive'), the counter electrode is
        # Li-metal; its OCP was already injected by _inject_reference_electrode_params.
        # Skip negative-electrode OCP overrides here to avoid overwriting it.
        if sim_params.working_electrode != "positive":
            if len(neg_mats_with_ocv) > 1:
                blend_ocp = _build_blend_ocp_interpolant(
                    cell_params.negative_electrode_active_materials,
                    pb,
                    ocp_mode=sim_params.negative_electrode_ocv_model,
                )
                if blend_ocp is not None:
                    neg_ocp_func = blend_ocp
                    param.update(
                        {"Negative electrode OCP [V]": blend_ocp},
                        check_already_exists=False,
                    )
            elif len(neg_mats_with_ocv) == 1:
                neg_ocp_func = _build_ocp_interpolant(
                    neg_mats_with_ocv[0],
                    pb,
                    ocp_mode=sim_params.negative_electrode_ocv_model,
                )
                param.update(
                    {"Negative electrode OCP [V]": neg_ocp_func},
                    check_already_exists=False,
                )
                if neg_mats_with_ocv[0].max_lithium_conc_mol_m3 is not None:
                    param.update(
                        {
                            "Maximum concentration in negative electrode [mol.m-3]": neg_mats_with_ocv[
                                0
                            ].max_lithium_conc_mol_m3
                        },
                        check_already_exists=False,
                    )
        sto_windows = cell_equilibrium_kpis.stoichiometry_windows
        # If stoichiometry windows are provided, use them to set OCV0/OCV100
        if (
            sto_windows is not None
            and pos_ocp_func is not None
            and neg_ocp_func is not None
        ):
            try:
                ocv0 = float(pos_ocp_func(sto_windows.sto_ca0)) - float(
                    neg_ocp_func(sto_windows.sto_an0)
                )
                ocv100 = float(pos_ocp_func(sto_windows.sto_ca100)) - float(
                    neg_ocp_func(sto_windows.sto_an100)
                )
                param.update(
                    {
                        "Open-circuit voltage at 0% SOC [V]": ocv0,
                        "Open-circuit voltage at 100% SOC [V]": ocv100,
                    },
                    check_already_exists=False,
                )
                logger.warning(
                    "  OCV from stoichiometry windows: OCV0=%.3f V, OCV100=%.3f V",
                    ocv0,
                    ocv100,
                )
            except (TypeError, ValueError) as e:
                logger.warning(
                    "Could not compute OCV from stoichiometry windows: %s", e
                )
        # If no OCV data provided, we'll rely on calibration to estimate OCV at 0% and 100% SOC.
        if cell_params.cell_ocv is None:
            logger.warning(
                "No OCV data provided for either electrode; will rely on calibration to estimate OCV at 0%% and 100%% SOC."
            )
        if cell_params.cell_ocv is not None:
            soc_arr = np.array(cell_params.cell_ocv.soc_pct)
            dch_arr = np.array(cell_params.cell_ocv.dch_ocv_V)
            if len(soc_arr) >= 2:
                idx_0 = np.argmin(np.abs(soc_arr - 0.0))
                idx_1 = np.argmin(np.abs(soc_arr - 100.0))
                param.update(
                    {
                        "Open-circuit voltage at 0% SOC [V]": float(dch_arr[idx_0]),
                        "Open-circuit voltage at 100% SOC [V]": float(dch_arr[idx_1]),
                    },
                    check_already_exists=False,
                )

    return param


# =============================================================================
# SECTION 4 — SIMULATION RUNNERS
# =============================================================================


def _estimate_ocv_bounds(
    cell_params: CellParametersInput,
    param: pybamm.ParameterValues,
    cell_equilibrium_kpis: CellEquilibriumKPIs,
    sim_params: SimulationParameters,
) -> pybamm.ParameterValues:
    """Estimate OCV at 0% and 100% SOC via two 1-second rest simulations.

    Replaces the iterative charge/discharge calibration loop. Electrode dimensions
    and nominal capacity are taken directly from the equilibrium calculation
    (mass loading * specific capacity * AM fraction * area) — no width adjustment.
    """
    pb = pybamm
    param = param.copy()

    # Remove any pre-existing "Open-circuit voltage at N% SOC [V]" keys inherited
    # from the base parameter set (e.g. Chen2020 defaults).  This ensures the
    # returned params only contain OCV bounds that were *actually estimated* for
    # this cell — a failed estimation will leave the key absent so the caller's
    # "if key in pybamm_params" check correctly yields None rather than a
    # misleading default from an unrelated cell geometry.
    for _ocv_key in (
        "Open-circuit voltage at 100% SOC [V]",
        "Open-circuit voltage at 0% SOC [V]",
    ):
        with contextlib.suppress(KeyError):
            del param[_ocv_key]

    sto_windows = cell_equilibrium_kpis.stoichiometry_windows
    model = _make_spme_model(
        cell_params, sim_params, "isothermal", set_anode_alias=False
    )
    experiment = pb.Experiment(["Rest for 1 seconds"], period="1 second")
    solver = pb.IDAKLUSolver(
        atol=sim_params.solver_atol,
        rtol=sim_params.solver_rtol,
        output_variables=["Time [s]", "Terminal voltage [V]"],
    )

    ocv_results: dict[str, float] = {}
    # Use 0.999 instead of 1.0 to avoid triggering the "Maximum voltage" event
    # at the upper voltage cutoff boundary when the initial state is fully charged.
    # Label "99.9%" reflects the actual SOC used so logs/comparisons are accurate.
    for target_soc, label in ((0.999, "99.9%"), (0.0, "0%")):
        sim = None
        sol = None
        try:
            p = param.copy()
            _apply_soc_to_param(
                p, sto_windows, target_soc, sim_params.working_electrode
            )
            sim = pb.Simulation(
                model,
                experiment=experiment,
                parameter_values=p,
                var_pts=sim_params.mesh_resolution,
            )
            sol = _solve_sim(sim, solver, sto_windows, target_soc)
            v = sol["Terminal voltage [V]"].entries
            if v.size > 0:
                ocv = float(v.flatten()[-1])
                ocv_results[label] = ocv
                logger.info("OCV at %s SOC: %.4f V", label, ocv)
        except Exception as exc:
            logger.warning("OCV estimation failed at SOC=%s: %s", label, exc)
        finally:
            del sim, sol
            gc.collect()

    updates: dict[str, float] = {}
    if "99.9%" in ocv_results:
        updates["Open-circuit voltage at 100% SOC [V]"] = ocv_results["99.9%"]
    if "0%" in ocv_results:
        updates["Open-circuit voltage at 0% SOC [V]"] = ocv_results["0%"]
    if updates:
        param.update(updates, check_already_exists=False)
    else:
        logger.warning(
            "OCV estimation failed at both 0%% and 100%% SOC; "
            "reported ocv100_V / ocv0_V will use the parameter-set defaults, "
            "which may be inaccurate for this cell geometry."
        )

    return param


def _run_c3_discharge_pybamm(
    cell_params: CellParametersInput,
    cell_equilibrium_kpis: CellEquilibriumKPIs,
    pybamm_params: pybamm.ParameterValues,
    sim_params: SimulationParameters,
    thermal_model: str = "isothermal",
) -> tuple[float | None, float | None, TimeseriesOutput | None]:
    """C/3 discharge. Returns (capacity_Ah, energy_Wh, timeseries) or (None, None, None) on failure."""
    pb = pybamm
    nominal_capacity_Ah = pybamm_params["Nominal cell capacity [A.h]"]
    upper_V = sim_params.upper_voltage_cutoff_V
    lower_V = sim_params.lower_voltage_cutoff_V
    sto_windows = cell_equilibrium_kpis.stoichiometry_windows

    c3 = nominal_capacity_Ah / 3.0
    experiment = pb.Experiment(
        [
            "Rest for 1 second",
            f"Charge at {c3} A until {upper_V} V",
            f"Hold at {upper_V} V for 1 hour or until C/50",
            "Rest for 10 seconds",
            f"Discharge at {c3} A until {lower_V} V",
        ],
        period="60 second",
    )
    model = _make_spme_model(cell_params, sim_params, thermal_model)

    # Start just below 100% so Charge-until-upper_V event is positive; 100% can hit upper_V and error
    target_soc = 0.5
    solve_param = pybamm_params.copy()
    _apply_soc_to_param(
        solve_param, sto_windows, target_soc, sim_params.working_electrode
    )

    sim = pb.Simulation(
        model,
        experiment=experiment,
        parameter_values=solve_param,
        var_pts=sim_params.mesh_resolution,
    )
    solver = pb.IDAKLUSolver(
        atol=sim_params.solver_atol,
        rtol=sim_params.solver_rtol,
        output_variables=[
            "Time [s]",
            "Terminal voltage [V]",
            "Current [A]",
            "Discharge capacity [A.h]",
            "Cell temperature [K]",
            "Discharge energy [W.h]",
            "Power [W]",
            *(
                ["Anode potential [V]"]
                if sim_params.working_electrode != "positive"
                else []
            ),
        ],
    )
    try:
        sol = _solve_sim(sim, solver, sto_windows, target_soc)
    except pb.SolverError as e:
        logger.warning("C/3 discharge solver error: %s", e)
        return (None, None, None)
    finally:
        del sim
        gc.collect()

    try:
        # Extract discharge cycle
        if not hasattr(sol, "cycles") or len(sol.cycles) < 3:
            logger.warning("C/3 discharge: missing cycles in solution")
            return (None, None, None)

        # The discharge step is usually the last step in the last cycle
        # or we can find it by looking for the one with negative current
        dch_cycle = sol.cycles[-1]

        # In PyBaMM Experiments, cycles contain steps.
        # Experiment steps: [Rest, Charge, Hold, Rest, Discharge]
        # sol.cycles[-1] should be the last cycle containing these steps.
        # Robustly identify discharge step by checking multiple criteria:
        # 1. Positive current (PyBaMM sign convention: discharge is > 0)
        # 2. Decreasing voltage (discharge characteristic)
        # 3. Significant discharge capacity gain
        # This avoids false positives from Rest steps (small current drift) or Hold steps (CV charging)
        dch_step = None
        steps = getattr(dch_cycle, "steps", None)
        if not steps or not hasattr(steps, "__iter__"):
            logger.warning(
                "C/3 discharge: dch_cycle.steps is missing or not iterable (got %r)",
                steps,
            )
            return (None, None, None)

        for step in reversed(list(steps)):
            try:
                entries = step["Current [A]"].entries
                if not hasattr(entries, "__len__") or len(entries) == 0:
                    continue
                if float(np.mean(entries)) > 0.01:  # discharge is positive in PyBaMM
                    dch_step = step
                    break
            except (KeyError, TypeError, AttributeError):
                continue

        if dch_step is None:
            logger.warning("C/3 discharge: could not identify discharge step")
            return (None, None, None)

        t = dch_step["Time [s]"].entries.astype(float)

        def _get_entries(var_name, target_t):
            vals = dch_step[var_name].entries.astype(float)
            # Handle spatial variables by averaging over spatial dimensions (all except last/time axis)
            if vals.ndim > 1:
                vals = np.mean(vals, axis=tuple(range(vals.ndim - 1)))

            if len(vals) == len(target_t):
                return vals
            # Fallback: interpolate to target_t
            return np.interp(
                target_t, dch_step["Time [s]"].entries.astype(float).flatten(), vals
            )

        v = _get_entries("Terminal voltage [V]", t)
        i = _get_entries("Current [A]", t)
        cap = _get_entries("Discharge capacity [A.h]", t)
        temp = _get_entries("Cell temperature [K]", t)
        energy = _get_entries("Discharge energy [W.h]", t)
        # Extract power from PyBaMM solution (or calculate from v * i if not available)
        try:
            power = _get_entries("Power [W]", t)
        except (KeyError, TypeError):
            # Fallback: calculate power from voltage and current
            power = v * i
        # Extract anode potential from PyBaMM solution
        try:
            anode_potential_V = _get_entries("Anode potential [V]", t)
        except (KeyError, TypeError):
            # Fallback: try alternative variable name
            try:
                anode_potential_V = _get_entries(
                    "Negative electrode surface potential difference at separator interface [V]",
                    t,
                )
            except (KeyError, TypeError):
                logger.warning(
                    "Anode potential not available in PyBaMM solution, using zeros"
                )
                anode_potential_V = np.zeros_like(t)

        cap_Ah = float(cap[-1] - cap[0])
        en_Wh = float(energy[-1] - energy[0])

        cap = cap - cap[0]
        energy = energy - energy[0]
        t = t - t[0]
        # Verify all arrays have the same length before stacking
        arrays = {
            "time": t,
            "voltage": v,
            "current": i,
            "temperature": temp,
            "capacity": cap,
            "energy": energy,
            "power": power,
            "anode_potential": anode_potential_V,
        }
        lengths = {name: len(arr) for name, arr in arrays.items()}
        if len(set(lengths.values())) > 1:
            logger.error("Array length mismatch in C/3 discharge data: %s", lengths)
            raise ValueError(
                f"Array length mismatch: {lengths}. Cannot construct timeseries."
            )

        # Subsample using RDP
        # Column order matches TimeseriesOutput: time_s, voltage_V, current_A, temperature_K,
        # capacity_Ah, energy_Wh, power_W, anode_potential_V
        eps = sim_params.timeseries_rdp_epsilon
        full_data = np.column_stack(
            (t, v, i, temp, cap, energy, power, anode_potential_V)
        )

        # Verify full_data has expected shape (N, 8)
        if full_data.shape[1] != 8:
            logger.error(
                "full_data has %d columns, expected 8. Shape: %s",
                full_data.shape[1],
                full_data.shape,
            )
            raise ValueError(f"full_data has {full_data.shape[1]} columns, expected 8")

        subsampled_data = _rdp_subsample(full_data, eps)

        # Verify subsampled_data still has 8 columns
        if subsampled_data.shape[1] != 8:
            logger.error(
                "subsampled_data has %d columns after RDP, expected 8. Shape: %s",
                subsampled_data.shape[1],
                subsampled_data.shape,
            )
            raise ValueError(
                f"subsampled_data has {subsampled_data.shape[1]} columns after RDP, expected 8"
            )

        ts = TimeseriesOutput(
            time_s=subsampled_data[:, 0].tolist(),
            voltage_V=subsampled_data[:, 1].tolist(),
            current_A=subsampled_data[:, 2].tolist(),
            temperature_K=subsampled_data[:, 3].tolist(),
            capacity_Ah=subsampled_data[:, 4].tolist(),
            energy_Wh=subsampled_data[:, 5].tolist(),
            power_W=subsampled_data[:, 6].tolist(),
            anode_potential_V=subsampled_data[:, 7].tolist(),
        )

        return (cap_Ah, en_Wh, ts)

    except (KeyError, TypeError, IndexError, ValueError) as e:
        logger.warning("C/3 extraction failed: %s", e)
        # Preserve capacity and energy values even if timeseries construction fails
        cap_Ah_preserved = cap_Ah if "cap_Ah" in locals() else None
        en_Wh_preserved = en_Wh if "en_Wh" in locals() else None
        if "sol" in locals():
            del sol
        gc.collect()
        return (cap_Ah_preserved, en_Wh_preserved, None)


def _run_dcir_10s_pybamm(
    cell_params: CellParametersInput,
    cell_equilibrium_kpis: CellEquilibriumKPIs,
    pybamm_params: pybamm.ParameterValues,
    sim_params: SimulationParameters,
    thermal_model: str = "isothermal",
) -> float | None:
    """DCIR measurement. Returns mOhm or None on failure."""
    pb = pybamm
    try:
        capacity_Ah = pybamm_params["Nominal cell capacity [A.h]"]
    except (KeyError, TypeError):
        capacity_Ah = cell_equilibrium_kpis.cell_theoretical_capacity_Ah
    dcir = sim_params.rpt_dcir
    direction = (dcir.dcir_direction or "discharge").lower()
    # Sign convention: discharge is > 0, charge is < 0 (matches prep_simulation_experiment)
    # Discharge uses positive current, charge uses negative current

    # Set up experiment
    i_pulse = capacity_Ah * dcir.dcir_c_rate * (-1 if direction == "charge" else 1)
    rest_s = dcir.dcir_rest_s
    rest_str = f"Rest for {rest_s} second{'s' if rest_s != 1 else ''}"

    experiment = pb.Experiment(
        [
            rest_str,
            (pb.step.current(i_pulse, duration=dcir.dcir_pulse_duration_s),),
        ],
        period="0.1 second",
    )

    model = _make_spme_model(
        cell_params, sim_params, thermal_model, set_anode_alias=False
    )

    target_soc = dcir.dcir_soc_pct / 100.0
    solve_param = pybamm_params.copy()
    sto_windows = cell_equilibrium_kpis.stoichiometry_windows
    _apply_soc_to_param(
        solve_param, sto_windows, target_soc, sim_params.working_electrode
    )
    ambient_temperature_K = (
        dcir.dcir_temperature_K if dcir.dcir_temperature_K is not None else 298.15
    )
    solve_param.update(
        {
            "Ambient temperature [K]": ambient_temperature_K,
            "Initial temperature [K]": ambient_temperature_K,
        },
        check_already_exists=False,
    )

    # Set up simulation
    sim = pb.Simulation(
        model,
        experiment=experiment,
        parameter_values=solve_param,
        var_pts=sim_params.mesh_resolution,
    )
    solver = pb.IDAKLUSolver(
        atol=sim_params.solver_atol,
        rtol=sim_params.solver_rtol,
        output_variables=[
            "Terminal voltage [V]",
            "Current [A]",
            "Time [s]",
        ],
    )

    try:
        sol = _solve_sim(sim, solver, sto_windows, target_soc)
    except pb.SolverError as e:
        logger.warning("DCIR solver error: %s", e)
        if "sol" in locals():
            del sol
        if "sim" in locals():
            del sim
        gc.collect()
        return None
    finally:
        if "sim" in locals():
            del sim
        gc.collect()

    # Extract data
    try:
        v_arr = sol["Terminal voltage [V]"].entries
        i_arr = sol["Current [A]"].entries
        t_arr = sol["Time [s]"].entries

        # Baseline: last sample where t <= rest_end_t (pre-pulse, after relaxation).
        # argmin(|t - boundary|) can pick the first pulse sample when the boundary
        # aligns with the solver grid; searchsorted with side="right" always returns
        # the insertion point so ins-1 is the last index strictly at or before the boundary.
        rest_s = dcir.dcir_rest_s
        t0 = t_arr[0]
        rest_end_t = t0 + rest_s
        pulse_end_t = rest_end_t + dcir.dcir_pulse_duration_s

        t_arr_np = np.array(t_arr)
        n_pts = len(v_arr)
        ins = int(np.searchsorted(t_arr_np, rest_end_t, side="right"))
        baseline_idx = min(max(0, ins - 1), n_pts - 1)
        ins = int(np.searchsorted(t_arr_np, pulse_end_t, side="right"))
        pulse_idx = min(max(0, ins - 1), n_pts - 1)

        v_0 = float(v_arr[baseline_idx])
        v_end = float(v_arr[pulse_idx])
        i_end = float(i_arr[pulse_idx])
        if abs(i_end) < 1e-6:
            logger.warning("DCIR: current too small at pulse end")
            return None
        dcir_ohm = (v_0 - v_end) / abs(i_end)
        return abs(dcir_ohm * 1000)
    except (KeyError, TypeError, IndexError) as e:
        logger.warning("DCIR extraction failed: %s", e)
        return None


# =============================================================================
# SECTION 5 — EXPERIMENT ENGINE
# =============================================================================


def _build_experiment_terminations(
    terminations: list[ExperimentTermination],
) -> list:
    """Convert ExperimentTermination items to pybamm CustomTermination objects."""
    pb = pybamm
    result = []
    for term in terminations:
        threshold = term.threshold
        comparison = term.comparison
        var_name = term.variable
        term_name = (
            term.name
            or f"{var_name} {'<' if comparison == 'less_than' else '>'} {threshold}"
        )
        if comparison == "less_than":

            def _make_lt(v: str, thr: float):
                # Return a PyBaMM symbolic expression; event fires when it becomes negative.
                # fires when variables[v] < thr  →  variables[v] - thr < 0
                def _func(variables: dict):
                    return variables[v] - thr

                return _func

            func = _make_lt(var_name, threshold)
        else:

            def _make_gt(v: str, thr: float):
                # fires when variables[v] > thr  →  thr - variables[v] < 0
                def _func(variables: dict):
                    return thr - variables[v]

                return _func

            func = _make_gt(var_name, threshold)
        result.append(pb.step.CustomTermination(term_name, func))
    return result


def _build_pb_step(
    step: str | DriveStepConfig | EISStepConfig,
    terminations: list,
) -> object:
    """Convert one step config to a PyBaMM step object.

    String steps pass through unchanged (with optional terminations attached).
    DriveStepConfig steps become ``pybamm.step.current/power/c_rate`` objects.
    EISStepConfig must NOT be passed here - use the segmented execution path.
    """
    pb = pybamm
    if isinstance(step, str):
        parsed = pb.step.string(step)
        if terminations:
            parsed.termination.extend(terminations)
        return parsed
    if isinstance(step, DriveStepConfig):
        time_data = np.array(step.time_s, dtype=float)
        time_data = time_data - time_data[0]  # normalize to start at 0
        duration = float(time_data[-1])
        if step.current_A is not None:
            profile = np.array(step.current_A, dtype=float)
            data = np.column_stack([time_data, profile])
            pb_step = pb.step.current(data, duration=duration)
        elif step.power_W is not None:
            profile = np.array(step.power_W, dtype=float)
            data = np.column_stack([time_data, profile])
            pb_step = pb.step.power(data, duration=duration)
        else:
            profile = np.array(step.c_rate, dtype=float)  # type: ignore[arg-type]
            data = np.column_stack([time_data, profile])
            pb_step = pb.step.c_rate(data, duration=duration)
        if terminations:
            pb_step.termination.extend(terminations)
        return pb_step
    raise TypeError(f"EISStepConfig cannot be used in fast-path: {step!r}")


def _extract_step_timeseries(
    step_sol: object,
    time_offset: float,
) -> tuple[np.ndarray, ...] | None:
    """Extract arrays from one step solution sub-object.

    Returns ``(t_s, v_s, i_s, cap_s, temp_s, energy_s, power_s, anode_s)``
    with capacity and energy zeroed at step start, or ``None`` on failure.
    """
    try:
        t_raw = step_sol["Time [s]"].entries.astype(float)  # type: ignore[index]
        if len(t_raw) == 0:
            return None

        def _ext(var: str, t_ref: np.ndarray, _s: object = step_sol) -> np.ndarray:
            vals = _s[var].entries.astype(float)  # type: ignore[index]
            if vals.ndim > 1:
                vals = np.mean(vals, axis=tuple(range(vals.ndim - 1)))
            if len(vals) != len(t_ref):
                raise ValueError(
                    f"Length mismatch '{var}': {len(vals)} vs {len(t_ref)}"
                )
            return vals

        t_s = t_raw - t_raw[0] + time_offset
        v_s = _ext("Terminal voltage [V]", t_raw)
        i_s = _ext("Current [A]", t_raw)
        cap_s = _ext("Throughput capacity [A.h]", t_raw)
        temp_s = _ext("Cell temperature [K]", t_raw)
        energy_s = _ext("Throughput energy [W.h]", t_raw)
        try:
            power_s = _ext("Power [W]", t_raw)
        except (KeyError, TypeError, ValueError):
            power_s = v_s * i_s
        try:
            anode_s = _ext("Anode potential [V]", t_raw)
        except (KeyError, TypeError, ValueError):
            anode_s = np.zeros_like(t_raw)

        # Zero at step start; caller (_accumulate_step_data) is responsible
        # for adding cumulative offsets so the concatenated timeseries is
        # monotonically increasing across step boundaries.
        cap_s = cap_s - cap_s[0]
        energy_s = energy_s - energy_s[0]
        return t_s, v_s, i_s, cap_s, temp_s, energy_s, power_s, anode_s
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        logger.debug("Step timeseries extraction failed: %s", exc)
        return None


def _run_eis_step(
    eis_config: EISStepConfig,
    cell_params: CellParametersInput,
    pybamm_params: pybamm.ParameterValues,
    sim_params: SimulationParameters,
    sto_windows: StoichiometryWindows | None,
    current_soc: float,
    ambient_K: float,
    thermal_model: str,
    cycle_number: int,
    step_index: int,
    experiment_warnings: list[str] | None = None,
) -> EISOutput | None:
    """Run a single EIS frequency sweep initialised from *current_soc*.

    This is an SOC-based snapshot: the electrode concentrations are set via
    ``_apply_soc_to_param`` rather than continuing from the terminal state of
    the preceding time-domain simulation.  This is intentional — pybammeis
    does not accept a ``starting_solution`` and always initialises from the
    parameter values.  The returned :class:`EISOutput` records the SOC at
    which the measurement was taken so callers can correlate it with the
    timeseries.

    Requires ``pybammeis``.  Returns :class:`EISOutput` or ``None`` on
    failure or if the library is unavailable.
    """
    if not _check_pybammeis():
        return None

    if sto_windows is None:
        msg = (
            f"EIS step skipped (cycle {cycle_number}, step {step_index}): "
            "stoichiometry windows are unavailable so the initial electrode state "
            "cannot be set. Run the equilibrium calculation to enable EIS."
        )
        logger.warning(msg)
        if experiment_warnings is not None:
            experiment_warnings.append(msg)
        return None

    import pybammeis  # type: ignore[import]

    pb = pybamm
    eis_model = None
    eis_sim = None
    try:
        eis_options = _get_spme_model_options(
            cell_params,
            thermal=thermal_model,
            working_electrode=sim_params.working_electrode,
            particle_diffusion_model=sim_params.particle_diffusion_model,
            negative_electrode_phases=sim_params.negative_electrode_phases,
            positive_electrode_phases=sim_params.positive_electrode_phases,
            enable_double_layer_capacitance=sim_params.enable_double_layer_capacitance,
        )
        eis_options["surface form"] = "differential"
        eis_model = pb.lithium_ion.SPMe(options=eis_options)

        eis_param = pybamm_params.copy()
        eis_param.update(
            {
                "Ambient temperature [K]": ambient_K,
                "Initial temperature [K]": ambient_K,
            },
            check_already_exists=False,
        )
        # Set battery state via concentration overrides from equilibrium sto_windows.
        # pybammeis.EISSimulation.solve() takes (frequencies, method, inputs) —
        # it does not accept initial_soc.
        _apply_soc_to_param(
            eis_param, sto_windows, current_soc, sim_params.working_electrode
        )
        eis_sim = pybammeis.EISSimulation(
            eis_model,
            parameter_values=eis_param,
            var_pts=sim_params.mesh_resolution,
        )
        freqs = np.logspace(
            np.log10(eis_config.freq_min_hz),
            np.log10(eis_config.freq_max_hz),
            eis_config.n_frequencies,
        )
        eis_sim.solve(freqs, method=eis_config.method)

        z: np.ndarray = np.asarray(eis_sim.solution, dtype=complex)
        return EISOutput(
            frequencies_hz=freqs.tolist(),
            z_real_ohm=z.real.tolist(),
            z_imag_ohm=z.imag.tolist(),
            z_magnitude_ohm=np.abs(z).tolist(),
            z_phase_deg=np.degrees(np.angle(z)).tolist(),
            cycle_number=cycle_number,
            step_index=step_index,
            soc_pct=round(current_soc * 100.0, 2),
        )
    except Exception as exc:
        logger.warning(
            "EIS step failed (cycle %d, step %d): %s",
            cycle_number,
            step_index,
            exc,
        )
        return None
    finally:
        del eis_model, eis_sim
        gc.collect()


def _run_sub_experiment(
    pending_steps: list,
    model: object,
    solve_param: pybamm.ParameterValues,
    current_soc: float,
    config: ExperimentConfig,
    solver: object,
    sim_params: SimulationParameters,
    time_offset: float,
    label: str,
    cycle_idx: int,
    starting_solution: object | None = None,
) -> tuple[bool, list[tuple], float, float, object | None]:
    """Execute a PyBaMM sub-experiment for *pending_steps* within one cycle.

    Returns ``(success, step_data_list, new_soc, new_time_offset, solution)``.
    Each element of *step_data_list* is a tuple of eight arrays:
    ``(t_s, v_s, i_s, cap_s, temp_s, energy_s, power_s, anode_s)``.
    The returned *solution* should be passed as *starting_solution* to the
    next call to preserve battery internal state across EIS steps and cycles.
    Returns ``(False, [], current_soc, time_offset, None)`` on solver failure.

    When *starting_solution* is provided the simulation continues from the
    terminal state of that solution rather than reinitialising from *current_soc*.
    This preserves the full internal state (electrode concentrations, potentials,
    temperature) across segment boundaries, avoiding discontinuities in the
    timeseries and incorrect behaviour when EIS steps interrupt the cycle.
    When *starting_solution* is None (first segment of the first cycle) the
    simulation is initialised via PyBaMM's ``initial_soc`` mechanism instead.
    """
    pb = pybamm
    if not pending_steps:
        return True, [], current_soc, time_offset, starting_solution

    sub_param = solve_param.copy()
    sub_exp = pb.Experiment(
        [tuple(pending_steps)],
        period=f"{config.period_s} second",
    )
    sub_sim = pb.Simulation(
        model,
        experiment=sub_exp,
        parameter_values=sub_param,
        var_pts=sim_params.mesh_resolution,
    )
    try:
        if starting_solution is not None:
            sub_sol = sub_sim.solve(starting_solution=starting_solution, solver=solver)
        else:
            # First segment: no prior solution, initialise from SOC.
            # PyBaMM's initial_soc maps the SOC through the calibrated OCP
            # functions and is always consistent with the active parameter set.
            sub_sol = sub_sim.solve(initial_soc=current_soc, solver=solver)
    except pb.SolverError as exc:
        logger.warning(
            "Experiment segmented solver error (%s) cycle %d: %s",
            label,
            cycle_idx + 1,
            exc,
        )
        return False, [], current_soc, time_offset, None
    finally:
        del sub_sim
        gc.collect()

    if not hasattr(sub_sol, "cycles") or not sub_sol.cycles:
        return True, [], current_soc, time_offset, sub_sol

    sub_cycle = sub_sol.cycles[0]
    sub_steps = list(getattr(sub_cycle, "steps", []))
    if not sub_steps:
        sub_steps = [sub_cycle]

    step_data_list: list[tuple] = []
    seg_charge_Ah = 0.0
    seg_discharge_Ah = 0.0
    cur_offset = time_offset

    for step_sol in sub_steps:
        extracted = _extract_step_timeseries(step_sol, cur_offset)
        if extracted is None:
            continue
        t_s, v_s, i_s, cap_s, temp_s, energy_s, power_s, anode_s = extracted
        # Use signed current integral for correct SOC tracking on drive-cycle
        # steps where current changes sign within a single step.  Positive
        # current = discharge (PyBaMM convention).
        seg_discharge_Ah += float(np.trapz(np.maximum(i_s, 0.0), t_s) / 3600.0)
        seg_charge_Ah += float(np.trapz(np.maximum(-i_s, 0.0), t_s) / 3600.0)
        step_data_list.append(
            (t_s, v_s, i_s, cap_s, temp_s, energy_s, power_s, anode_s)
        )
        cur_offset = float(t_s[-1])

    nominal_Ah = float(sub_param.get("Nominal cell capacity [A.h]", 1.0))
    soc_delta = (seg_discharge_Ah - seg_charge_Ah) / nominal_Ah
    new_soc = float(np.clip(current_soc - soc_delta, 0.0, 1.0))
    return True, step_data_list, new_soc, cur_offset, sub_sol


def _make_cycle_accum() -> dict:
    """Return a fresh per-cycle accumulator dict."""
    return {
        "discharge_Ah": 0.0,
        "charge_Ah": 0.0,
        "cap_throughput_Ah": 0.0,
        "energy_throughput_Wh": 0.0,
        "v_min": None,
        "v_max": None,
        "temp_max": None,
    }


def _accumulate_step_data(
    sd: tuple,
    accum: dict,
    all_arrays: list,
) -> None:
    """Accumulate one step's timeseries into *accum* and *all_arrays*.

    *sd* is ``(t_s, v_s, i_s, cap_s, temp_s, energy_s, power_s, anode_s)``.
    *accum* must be a dict from :func:`_make_cycle_accum`.
    *all_arrays* is ``[all_t, all_v, all_i, all_cap, all_temp, all_energy,
    all_power, all_anode]`` — each a list that is mutated in place.
    """
    t_s, v_s, i_s, _cap_s, temp_s, energy_s, power_s, anode_s = sd
    i_pos = np.clip(i_s, 0.0, None)
    i_neg = np.clip(i_s, None, 0.0)
    discharge_Ah = float(np.trapz(i_pos, t_s) / 3600.0)
    charge_Ah = float(-np.trapz(i_neg, t_s) / 3600.0)
    accum["discharge_Ah"] += discharge_Ah
    accum["charge_Ah"] += charge_Ah
    accum["cap_throughput_Ah"] += discharge_Ah + charge_Ah
    # Integrate |power| over time for true bidirectional energy throughput.
    # abs(energy_s[-1]) would give |net energy| for mixed-sign steps, which
    # undercounts when discharge and regen occur within the same step.
    accum["energy_throughput_Wh"] += float(np.trapz(np.abs(power_s), t_s) / 3600.0)
    v_min = float(np.min(v_s))
    v_max = float(np.max(v_s))
    temp_max = float(np.max(temp_s))
    accum["v_min"] = v_min if accum["v_min"] is None else min(accum["v_min"], v_min)
    accum["v_max"] = v_max if accum["v_max"] is None else max(accum["v_max"], v_max)
    accum["temp_max"] = (
        temp_max if accum["temp_max"] is None else max(accum["temp_max"], temp_max)
    )
    all_t, all_v, all_i, all_cap, all_temp, all_energy, all_power, all_anode = (
        all_arrays
    )
    # Skip the first sample only when joining with prior data (deduplicates the
    # shared boundary point between steps).  If the step has exactly one sample
    # do NOT skip it — it is still valid data and must be appended so callers
    # can correctly read the final voltage / capacity at that boundary.
    skip = 1 if all_t and len(t_s) > 1 else 0
    if len(t_s) == 0:
        return
    cap_s = sd[3]
    # Each step's cap/energy is zeroed at its own start by _extract_step_timeseries.
    # Add the cumulative total from all prior steps so the concatenated timeseries
    # is continuous rather than resetting to ~0 at every step boundary.
    cap_offset = float(all_cap[-1][-1]) if all_cap else 0.0
    energy_offset = float(all_energy[-1][-1]) if all_energy else 0.0
    all_t.append(t_s[skip:])
    all_v.append(v_s[skip:])
    all_i.append(i_s[skip:])
    all_cap.append(cap_s[skip:] + cap_offset)
    all_temp.append(temp_s[skip:])
    all_energy.append(energy_s[skip:] + energy_offset)
    all_power.append(power_s[skip:])
    all_anode.append(anode_s[skip:])


def _build_cycle_summary(
    cycle_idx: int, accum: dict, ce: float | None
) -> ExperimentCycleSummary:
    """Build an :class:`ExperimentCycleSummary` from a completed accumulator."""
    return ExperimentCycleSummary(
        cycle_number=cycle_idx + 1,
        discharge_capacity_Ah=round(accum["discharge_Ah"], 6),
        charge_capacity_Ah=round(accum["charge_Ah"], 6),
        coulombic_efficiency_pct=ce,
        total_capacity_throughput_Ah=round(accum["cap_throughput_Ah"], 6),
        total_energy_throughput_Wh=round(accum["energy_throughput_Wh"], 6),
        min_voltage_V=round(accum["v_min"], 4) if accum["v_min"] is not None else None,
        max_voltage_V=round(accum["v_max"], 4) if accum["v_max"] is not None else None,
        max_temperature_K=(
            round(accum["temp_max"], 4) if accum["temp_max"] is not None else None
        ),
    )


def _run_experiment_pybamm(
    cell_params: CellParametersInput,
    pybamm_params: pybamm.ParameterValues,
    sim_params: SimulationParameters,
    config: ExperimentConfig,
    sto_windows: StoichiometryWindows | None = None,
    thermal_model: str = "lumped",
) -> ExperimentOutput | None:
    """Run a multi-step / multi-cycle PyBaMM experiment defined by *config*.

    Supports three step types:

    - ``str`` - standard PyBaMM step string (e.g. "Discharge at 1C for 1 hour")
    - :class:`DriveStepConfig` - time-varying current, power, or C-rate profile
    - :class:`EISStepConfig` - frequency-domain impedance snapshot (requires pybammeis)

    **Fast path** (no EIS steps): all *n_cycles* cycles are built at once and
    solved in a single PyBaMM call.

    **Segmented path** (EIS steps present): each cycle is executed step-by-step.
    Non-EIS steps are grouped into PyBaMM sub-experiments; EIS steps call
    ``pybammeis.EISSimulation`` separately.  SOC is tracked between segments.

    Returns an :class:`ExperimentOutput` or ``None`` on solver failure.
    """
    pb = pybamm
    label = config.label
    initial_soc = config.initial_soc_pct / 100.0
    ambient_K = config.temperature_K

    custom_terminations = _build_experiment_terminations(config.terminations or [])
    has_eis = any(isinstance(s, EISStepConfig) for s in config.steps)

    experiment_warnings: list[str] = []
    if has_eis and not _check_pybammeis():
        experiment_warnings.append(
            "EIS steps were requested but 'pybammeis' is not installed in this "
            "environment — all EIS steps will be skipped and eis_measurements will "
            "be empty. Install pybammeis to enable impedance support."
        )
        # Treat as fast path: all EIS steps are no-ops, so there is no reason
        # to pay the per-cycle overhead of the segmented execution path.
        has_eis = False

    model = _make_spme_model(cell_params, sim_params, thermal_model)

    solve_param = pybamm_params.copy()
    solve_param.update(
        {"Ambient temperature [K]": ambient_K, "Initial temperature [K]": ambient_K},
        check_already_exists=False,
    )
    # Do NOT apply sto_windows concentration overrides here.  Both the fast path and
    # the segmented path initialize SOC via PyBaMM's initial_soc argument, which maps
    # through the calibrated OCP functions.  Applying pre-calibration sto_windows
    # overrides would produce inconsistent (and potentially unphysical) initial states.

    solver = pb.IDAKLUSolver(
        atol=sim_params.solver_atol,
        rtol=sim_params.solver_rtol,
        output_variables=[
            "Time [s]",
            "Terminal voltage [V]",
            "Current [A]",
            "Throughput capacity [A.h]",
            "Cell temperature [K]",
            "Throughput energy [W.h]",
            "Power [W]",
            *(
                ["Anode potential [V]"]
                if sim_params.working_electrode != "positive"
                else []
            ),
        ],
    )

    # Shared output accumulators.
    all_t: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    all_i: list[np.ndarray] = []
    all_cap: list[np.ndarray] = []
    all_temp: list[np.ndarray] = []
    all_energy: list[np.ndarray] = []
    all_power: list[np.ndarray] = []
    all_anode: list[np.ndarray] = []
    cycle_summaries: list[ExperimentCycleSummary] = []
    eis_measurements: list[EISOutput] = []
    time_offset = 0.0
    n_cycles_completed = 0

    if not has_eis:
        # ================================================================
        # FAST PATH: no EIS - build all cycles at once, single solver call.
        # ================================================================
        built_steps = [
            _build_pb_step(s, custom_terminations)
            for s in config.steps
            if not isinstance(s, EISStepConfig)
        ]
        if not built_steps:
            # All steps were EIS-only and pybammeis is unavailable — nothing to run.
            experiment_warnings.append(
                f"Experiment '{label}' contains only EIS steps but 'pybammeis' is "
                "not installed. No simulation was performed."
            )
            return ExperimentOutput(
                timeseries=TimeseriesOutput(
                    time_s=[],
                    voltage_V=[],
                    current_A=[],
                    temperature_K=[],
                    capacity_Ah=[],
                    energy_Wh=[],
                    power_W=[],
                    anode_potential_V=[],
                ),
                cycle_summaries=[],
                n_cycles_completed=0,
                eis_measurements=[],
                warnings=experiment_warnings,
            )
        cycle_tuple = tuple(built_steps)
        experiment = pb.Experiment(
            [cycle_tuple] * config.n_cycles,
            period=f"{config.period_s} second",
        )
        sim = pb.Simulation(
            model,
            experiment=experiment,
            parameter_values=solve_param,
            var_pts=sim_params.mesh_resolution,
        )
        try:
            sol = sim.solve(initial_soc=initial_soc, solver=solver)
        except pb.SolverError as exc:
            logger.warning("Experiment solver error (%s): %s", label, exc)
            return None
        finally:
            del sim
            gc.collect()

        if not hasattr(sol, "cycles") or not sol.cycles:
            logger.warning("Experiment (%s): no cycles in solution", label)
            return None

        n_steps_per_cycle = sum(
            1 for s in config.steps if not isinstance(s, EISStepConfig)
        )
        all_arrays = [
            all_t,
            all_v,
            all_i,
            all_cap,
            all_temp,
            all_energy,
            all_power,
            all_anode,
        ]

        for cycle_idx, cycle_sol in enumerate(sol.cycles):
            steps = list(getattr(cycle_sol, "steps", []))
            if not steps:
                steps = [cycle_sol]

            accum = _make_cycle_accum()
            for step_sol in steps[:n_steps_per_cycle]:
                extracted = _extract_step_timeseries(step_sol, time_offset)
                if extracted is None:
                    logger.warning(
                        "Experiment (%s) cycle %d: step extraction returned None",
                        label,
                        cycle_idx + 1,
                    )
                    continue
                _accumulate_step_data(extracted, accum, all_arrays)
                time_offset = float(extracted[0][-1])

            # Only count and summarise cycles where all configured steps ran.
            # Partial cycles (early termination) are included in the timeseries
            # but do not receive a summary entry.
            if len(steps) >= n_steps_per_cycle:
                n_cycles_completed += 1
                ce: float | None = None
                if accum["charge_Ah"] > 0:
                    ce = round(100.0 * accum["discharge_Ah"] / accum["charge_Ah"], 2)
                cycle_summaries.append(_build_cycle_summary(cycle_idx, accum, ce))

    else:
        # ================================================================
        # SEGMENTED PATH: EIS present - execute cycle-by-cycle.
        # ================================================================
        current_soc = initial_soc
        all_arrays = [
            all_t,
            all_v,
            all_i,
            all_cap,
            all_temp,
            all_energy,
            all_power,
            all_anode,
        ]
        # Thread the terminal solution across segments and cycles so each
        # sub-experiment continues from the exact internal state where the
        # previous one ended, preserving electrode concentrations, potentials,
        # and temperature rather than reinitialising from SOC alone.
        prev_solution: object | None = None

        for cycle_idx in range(config.n_cycles):
            # Snapshot array lengths before starting this cycle so we can
            # roll back any partially-appended data on solver failure.
            len_snapshot = [len(arr) for arr in all_arrays]
            eis_len_snapshot = len(eis_measurements)

            accum = _make_cycle_accum()
            pending_steps: list = []
            cycle_ok = True

            for step_idx, step in enumerate(config.steps):
                if isinstance(step, EISStepConfig):
                    # Flush pending time-domain steps first.
                    ok, step_data_list, current_soc, time_offset, prev_solution = (
                        _run_sub_experiment(
                            pending_steps,
                            model,
                            solve_param,
                            current_soc,
                            config,
                            solver,
                            sim_params,
                            time_offset,
                            label,
                            cycle_idx,
                            starting_solution=prev_solution,
                        )
                    )
                    pending_steps = []
                    if not ok:
                        cycle_ok = False
                        break
                    for sd in step_data_list:
                        _accumulate_step_data(sd, accum, all_arrays)
                    # Run EIS snapshot at current SOC.
                    eis_result = _run_eis_step(
                        step,
                        cell_params,
                        pybamm_params,
                        sim_params,
                        sto_windows,
                        current_soc,
                        ambient_K,
                        thermal_model,
                        cycle_number=cycle_idx + 1,
                        step_index=step_idx,
                        experiment_warnings=experiment_warnings,
                    )
                    if eis_result is not None:
                        eis_measurements.append(eis_result)
                else:
                    pending_steps.append(_build_pb_step(step, custom_terminations))

            # Flush any remaining pending steps at end of cycle.
            if cycle_ok and pending_steps:
                ok, step_data_list, current_soc, time_offset, prev_solution = (
                    _run_sub_experiment(
                        pending_steps,
                        model,
                        solve_param,
                        current_soc,
                        config,
                        solver,
                        sim_params,
                        time_offset,
                        label,
                        cycle_idx,
                        starting_solution=prev_solution,
                    )
                )
                if not ok:
                    cycle_ok = False
                else:
                    for sd in step_data_list:
                        _accumulate_step_data(sd, accum, all_arrays)

            if not cycle_ok:
                # Roll back any partial timeseries appended during this cycle
                # so the returned arrays are consistent with cycle_summaries
                # (which will not contain a summary for this failed cycle).
                for arr, snap_len in zip(all_arrays, len_snapshot, strict=True):
                    del arr[snap_len:]
                del eis_measurements[eis_len_snapshot:]
                experiment_warnings.append(
                    f"Solver failure in cycle {cycle_idx + 1}; output contains "
                    f"partial data from {n_cycles_completed} completed cycle(s)."
                )
                break

            n_cycles_completed += 1
            ce = (
                round(100.0 * accum["discharge_Ah"] / accum["charge_Ah"], 2)
                if accum["charge_Ah"] > 0
                else None
            )
            cycle_summaries.append(_build_cycle_summary(cycle_idx, accum, ce))

    # ================================================================
    # Assemble final output.
    # ================================================================
    if not all_t and not eis_measurements:
        logger.warning("Experiment (%s): no timeseries data extracted", label)
        return None

    if all_t:
        t_cat = np.concatenate(all_t)
        v_cat = np.concatenate(all_v)
        i_cat = np.concatenate(all_i)
        cap_cat = np.concatenate(all_cap)
        temp_cat = np.concatenate(all_temp)
        energy_cat = np.concatenate(all_energy)
        power_cat = np.concatenate(all_power)
        anode_cat = np.concatenate(all_anode)

        full_data = np.column_stack(
            (t_cat, v_cat, i_cat, temp_cat, cap_cat, energy_cat, power_cat, anode_cat)
        )
        subsampled = _rdp_subsample(full_data, sim_params.timeseries_rdp_epsilon)

        ts = TimeseriesOutput(
            time_s=subsampled[:, 0].tolist(),
            voltage_V=subsampled[:, 1].tolist(),
            current_A=subsampled[:, 2].tolist(),
            temperature_K=subsampled[:, 3].tolist(),
            capacity_Ah=subsampled[:, 4].tolist(),
            energy_Wh=subsampled[:, 5].tolist(),
            power_W=subsampled[:, 6].tolist(),
            anode_potential_V=subsampled[:, 7].tolist(),
        )
    else:
        # EIS-only experiment: no time-domain data, return empty timeseries.
        ts = TimeseriesOutput(
            time_s=[],
            voltage_V=[],
            current_A=[],
            temperature_K=[],
            capacity_Ah=[],
            energy_Wh=[],
            power_W=[],
            anode_potential_V=[],
        )
    cap_values = [
        s.total_capacity_throughput_Ah
        for s in cycle_summaries
        if s.total_capacity_throughput_Ah is not None
    ]
    energy_values = [
        s.total_energy_throughput_Wh
        for s in cycle_summaries
        if s.total_energy_throughput_Wh is not None
    ]
    return ExperimentOutput(
        timeseries=ts,
        cycle_summaries=cycle_summaries,
        n_cycles_completed=n_cycles_completed,
        total_capacity_throughput_Ah=round(sum(cap_values), 6) if cap_values else None,
        total_energy_throughput_Wh=round(sum(energy_values), 6)
        if energy_values
        else None,
        eis_measurements=eis_measurements,
        warnings=experiment_warnings,
    )


def _run_max_power_300s_pybamm(
    cell_params: CellParametersInput,
    cell_equilibrium_kpis: CellEquilibriumKPIs,
    pybamm_params: pybamm.ParameterValues,
    sim_params: SimulationParameters,
    nominal_energy_Wh: float | None = None,
) -> float | None:
    """Max power for given duration. Returns W (positive for discharge, negative for charge) or None."""
    pb = pybamm

    def anode_cutoff(variables):
        return (
            variables["Anode potential [V]"]
            - sim_params.anode_potential_safety_threshold_V
        )

    def temp_cutoff(variables):
        return (
            sim_params.temperature_safety_threshold_K
            - variables["Volume-averaged cell temperature [K]"]
        )

    model = _make_spme_model(cell_params, sim_params, "lumped")

    dur_s = sim_params.rpt_power.power_duration_s
    direction = (sim_params.rpt_power.power_direction or "discharge").lower()
    # Sign convention: discharge is > 0, charge is < 0 (matches prep_simulation_experiment)
    power_sign = 1 if direction == "discharge" else -1
    tol_W = 1.0  # convergence tolerance
    sto_windows = cell_equilibrium_kpis.stoichiometry_windows
    lower_V = sim_params.lower_voltage_cutoff_V
    energy_Wh = (
        nominal_energy_Wh
        if nominal_energy_Wh is not None
        else cell_equilibrium_kpis.cell_theoretical_energy_Wh
    )
    target_soc = sim_params.rpt_power.power_soc_pct / 100.0
    ambient_temperature_K = (
        sim_params.rpt_power.power_temperature_K
        if sim_params.rpt_power.power_temperature_K is not None
        else 298.15
    )
    pybamm_params.update(
        {
            "Ambient temperature [K]": ambient_temperature_K,
            "Initial temperature [K]": ambient_temperature_K,
        },
        check_already_exists=False,
    )

    def _try_power(power_magnitude_W: float) -> bool:
        """Return True if the cell sustains *power_magnitude_W* (magnitude) for *dur_s*.

        Applies power_sign to convert magnitude to signed power for PyBaMM.
        """
        solve_param = pybamm_params.copy()
        _apply_soc_to_param(
            solve_param, sto_windows, target_soc, sim_params.working_electrode
        )

        # Apply sign convention: discharge = positive, charge = negative
        signed_power_W = power_magnitude_W * power_sign

        experiment = pb.Experiment(
            [
                "Rest for 1 seconds",
                (
                    pb.step.power(
                        signed_power_W,
                        duration=dur_s,
                        termination=[
                            pb.step.CustomTermination(
                                "Anode potential cut-off [V]", anode_cutoff
                            ),
                            pb.step.CustomTermination(
                                "Temperature cut-off [K]", temp_cutoff
                            ),
                            f"{lower_V}V",
                        ],
                    ),
                ),
            ],
            period=10,
        )
        sim = pb.Simulation(
            model,
            experiment=experiment,
            parameter_values=solve_param,
            var_pts=sim_params.mesh_resolution,
        )
        solver = pb.IDAKLUSolver(
            atol=sim_params.solver_atol,
            rtol=sim_params.solver_rtol,
            output_variables=["Time [s]"],
        )
        try:
            if sto_windows is not None:
                sol = sim.solve(solver=solver)
            else:
                sol = sim.solve(initial_soc=target_soc, solver=solver)
        except pb.SolverError:
            # Clean up any partially constructed solution and simulation
            if "sol" in locals():
                del sol
            if "sim" in locals():
                del sim
            gc.collect()
            return False
        else:
            t = sol["Time [s]"].entries
            # Explicitly clean up solution and simulation before returning
            del sol
            del sim
            gc.collect()
            return len(t) > 0 and float(t[-1]) >= dur_s - 1

    # Phase 1: probe downward from max to find a power level that works
    hi = (
        10.0 * energy_Wh
        if energy_Wh and energy_Wh > 0
        else sim_params.rpt_power.power_level_W
    )
    lo = 0.0
    probe = hi
    step_n = 0
    while probe > tol_W:
        step_n += 1
        logger.warning("    Max-power probe %d: trying %.1f W", step_n, probe)
        if _try_power(probe):
            lo = probe
            logger.warning(
                "    Max-power probe %d: %.1f W — sustained, upper bound found",
                step_n,
                probe,
            )
            break
        logger.warning(
            "    Max-power probe %d: %.1f W — failed, halving", step_n, probe
        )
        hi = probe
        probe = probe / 2.0
    else:
        logger.warning("    Max-power: no feasible power level found")
        return None

    # Phase 2: binary search between lo and hi
    bisect_n = 0
    while hi - lo > tol_W:
        bisect_n += 1
        mid = (lo + hi) / 2.0
        logger.warning(
            "    Max-power bisect %d: trying %.1f W [%.1f-%.1f]", bisect_n, mid, lo, hi
        )
        if _try_power(mid):
            lo = mid
            logger.warning("    Max-power bisect %d: %.1f W — sustained", bisect_n, mid)
        else:
            hi = mid
            logger.warning("    Max-power bisect %d: %.1f W — failed", bisect_n, mid)

    logger.warning(
        "    Max-power search done: %.1f W (%d probe + %d bisect iterations)",
        lo,
        step_n,
        bisect_n,
    )
    # Return signed power: positive for discharge, negative for charge
    return lo * power_sign if lo > 0 else None


# =============================================================================
# SECTION 6 — PIPELINE & PUBLIC API
# =============================================================================


def _report(
    progress_callback: ProgressCallback | None,
    progress_pct: int,
    message: str,
) -> None:
    """Report progress to the job UI if callback is provided.

    Args:
        progress_callback: Optional callback function accepting (progress_pct: int, message: str)
        progress_pct: Progress percentage (0-100)
        message: Status message describing current operation
    """
    if progress_callback:
        progress_callback(progress_pct, message)


def _compute_performance_kpis(
    cell_params: CellParametersInput,
    sim_params: SimulationParameters,
    progress_callback: ProgressCallback | None = None,
) -> CellPerformanceKPIs | None:
    _report(progress_callback, 5, "Computing equilibrium KPIs")
    cell_equilibrium_kpis = _compute_equilibrium_kpis(
        cell_params, sim_params=sim_params
    )

    # Early guard: skip PyBaMM simulations if electrode area or cell volume is invalid
    # This prevents meaningless simulations when dimensions are missing/invalid
    positive_electrode_area_cm2 = _electrode_area_cm2(cell_params, is_positive=True)
    negative_electrode_area_cm2 = _electrode_area_cm2(cell_params, is_positive=False)
    cell_volume_L = cell_equilibrium_kpis.cell_volume_L

    if (
        positive_electrode_area_cm2 <= 0
        or negative_electrode_area_cm2 <= 0
        or cell_volume_L <= 0
    ):
        skip_reasons = []
        if positive_electrode_area_cm2 <= 0 or negative_electrode_area_cm2 <= 0:
            skip_reasons.append(
                f"electrode area is zero or negative "
                f"(pos={positive_electrode_area_cm2:.2f} cm², "
                f"neg={negative_electrode_area_cm2:.2f} cm²); "
                "check electrode width/height/diameter dimensions"
            )
        if cell_volume_L <= 0:
            ff = cell_params.form_factor.lower()
            if ff == "coin":
                skip_reasons.append(
                    "cell volume is zero — coin cell requires cell_diameter_mm "
                    "and cell_height_mm (cell thickness) to compute volume"
                )
            elif ff == "cylindrical":
                skip_reasons.append(
                    "cell volume is zero — cylindrical cell requires "
                    "cell_diameter_mm and cell_height_mm"
                )
            else:
                skip_reasons.append(
                    "cell volume is zero — provide cell_width_mm, "
                    "cell_height_mm, and cell_thickness_mm"
                )
        skip_msg = (
            "PyBaMM simulations skipped: " + "; ".join(skip_reasons) + ". "
            "Returning equilibrium-only KPIs."
        )
        logger.warning(skip_msg)
        _report(progress_callback, 100, "Skipped PyBaMM (invalid dimensions)")
        return CellPerformanceKPIs(
            **cell_equilibrium_kpis.model_dump(),
            cell_nominal_capacity_Ah=None,
            cell_nominal_energy_Wh=None,
            cell_nominal_gravimetric_energy_density_Wh_kg=None,
            cell_nominal_volumetric_energy_density_Wh_L=None,
            cell_nominal_max_discharge_power_300s_W=None,
            cell_nominal_gravimetric_power_density_W_kg=None,
            cell_nominal_volumetric_power_density_W_L=None,
            cell_dcir_10s_mohm=None,
            ocv100_V=None,
            ocv0_V=None,
            c3_discharge_timeseries=None,
            experiment_results=None,
        )

    _report(progress_callback, 15, "Building PyBaMM parameters")
    pybamm_params = _build_pybamm_params_from_equilibrium(
        cell_params, cell_equilibrium_kpis, sim_params
    )

    # Capacity is known geometrically: mass_loading * specific_capacity * AM_mass_frac * electrode_area.
    # This is already encoded in cell_equilibrium_kpis.cell_theoretical_capacity_Ah and in the
    # pybamm_params electrode dimensions (set from coating thickness + area).
    # No iterative electrode-width calibration is needed — just estimate OCV at 0% and 100% SOC
    # via two brief 1-second rest simulations, then proceed.
    thermal_model = "lumped" if sim_params.perform_rpt else "isothermal"
    if sim_params.perform_rpt:
        logger.info(
            "Using lumped thermal model for all simulations (RPT enabled; power test requires thermal effects)"
        )
    logger.info(
        "  [2/5] Geometry-based capacity: %.4f Ah "
        "(mass loading * specific capacity * AM fraction * electrode area)",
        cell_equilibrium_kpis.cell_theoretical_capacity_Ah,
    )
    _report(progress_callback, 30, "Estimating OCV bounds")
    pybamm_params = _estimate_ocv_bounds(
        cell_params, pybamm_params, cell_equilibrium_kpis, sim_params
    )
    _report(progress_callback, 40, "OCV bounds estimated")

    cap_Ah: float | None = None
    en_Wh: float | None = None
    ts_data: TimeseriesOutput | None = None
    _report(progress_callback, 50, "Running C/3 discharge simulation")
    logger.warning("  [3/5] Running C/3 discharge ...")
    cap_Ah, en_Wh, ts_data = _run_c3_discharge_pybamm(
        cell_params,
        cell_equilibrium_kpis,
        pybamm_params,
        sim_params,
        thermal_model=thermal_model,
    )
    logger.info(
        "  [3/5] C/3 discharge done (%.3f Ah, %.2f Wh)", cap_Ah or 0, en_Wh or 0
    )
    _report(progress_callback, 70, "C/3 discharge complete")

    # Run optional multi-cycle experiments
    experiment_results: dict[str, ExperimentOutput] = {}
    if sim_params.experiments:
        n_exps = len(sim_params.experiments)
        for exp_idx, exp_config in enumerate(sim_params.experiments):
            pct = 75 + min(int(5 * (exp_idx + 1) / max(n_exps, 1)), 4)
            _report(progress_callback, pct, f"Experiment: {exp_config.label}")
            logger.info("  [3c] Experiment: %s", exp_config.label)
            exp_out = _run_experiment_pybamm(
                cell_params,
                pybamm_params,
                sim_params,
                exp_config,
                sto_windows=cell_equilibrium_kpis.stoichiometry_windows,
                thermal_model=thermal_model,
            )
            if exp_out is not None:
                experiment_results[exp_config.label] = exp_out
            else:
                logger.warning("  [3c] Experiment failed: %s", exp_config.label)

    def _safe_div(num: float | None, denom: float) -> float | None:
        if num is None or denom <= 0:
            return None
        return round(num / denom, 4)

    dcir: float | None = None
    max_power_W: float | None = None
    if sim_params.perform_rpt:
        _report(progress_callback, 80, "Running DCIR test")
        logger.info("  [4/5] Running DCIR ...")
        dcir = _run_dcir_10s_pybamm(
            cell_params,
            cell_equilibrium_kpis,
            pybamm_params,
            sim_params,
            thermal_model=thermal_model,
        )
        logger.info("  [4/5] DCIR done (%.2f mOhm)", dcir or 0)
        _report(progress_callback, 85, "DCIR test complete")

        _report(progress_callback, 90, "Running max power test")
        logger.info("  [5/5] Running max power test ...")
        max_power_W = _run_max_power_300s_pybamm(
            cell_params,
            cell_equilibrium_kpis,
            pybamm_params,
            sim_params,
            nominal_energy_Wh=en_Wh,
        )
        logger.info("  [5/5] Max power test done (%.1f W)", max_power_W or 0)
        _report(progress_callback, 95, "Max power test complete")
    else:
        logger.info("  [4/5] Skipping DCIR test (RPT disabled)")
        logger.info("  [5/5] Skipping max power test (RPT disabled)")

    _report(progress_callback, 98, "Extracting results")
    logger.info("  [5/5] PyBaMM pipeline: complete (%s)", cell_params.form_factor)

    out = CellPerformanceKPIs(
        **cell_equilibrium_kpis.model_dump(),
        cell_nominal_capacity_Ah=round(cap_Ah, 4) if cap_Ah else None,
        cell_nominal_energy_Wh=round(en_Wh, 4) if en_Wh else None,
        cell_nominal_gravimetric_energy_density_Wh_kg=_safe_div(
            en_Wh, cell_equilibrium_kpis.cell_mass_g / 1000.0
        ),
        cell_nominal_volumetric_energy_density_Wh_L=_safe_div(
            en_Wh, cell_equilibrium_kpis.cell_volume_L
        ),
        cell_nominal_max_discharge_power_300s_W=(
            round(abs(max_power_W), 2) if max_power_W else None
        ),
        cell_nominal_gravimetric_power_density_W_kg=_safe_div(
            abs(max_power_W) if max_power_W else None,
            cell_equilibrium_kpis.cell_mass_g / 1000.0,
        ),
        cell_nominal_volumetric_power_density_W_L=_safe_div(
            abs(max_power_W) if max_power_W else None,
            cell_equilibrium_kpis.cell_volume_L,
        ),
        cell_dcir_10s_mohm=round(dcir, 2) if dcir else None,
        ocv100_V=(
            round(pybamm_params["Open-circuit voltage at 100% SOC [V]"], 4)
            if "Open-circuit voltage at 100% SOC [V]" in pybamm_params
            else None
        ),
        ocv0_V=(
            round(pybamm_params["Open-circuit voltage at 0% SOC [V]"], 4)
            if "Open-circuit voltage at 0% SOC [V]" in pybamm_params
            else None
        ),
        c3_discharge_timeseries=ts_data,
        experiment_results=experiment_results if sim_params.experiments else None,
    )  # type: ignore[return-value]

    return out


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================


def calculate_cell_performance(
    parameters: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Compute cell performance KPIs.

    Wire boundary: accepts dict from service layer, parses to Pydantic, returns .model_dump().
    Progress callback is optional and used for UI progress reporting.

    Returns:
        Dictionary of cell performance KPIs, or empty dict if computation fails

    Raises:
        ValidationError: If input parameters are invalid
    """
    parsed = CellPerformanceInput.model_validate(parameters)
    result = _compute_performance_kpis(
        cell_params=parsed.cell_parameters,
        sim_params=parsed.simulation_parameters,
        progress_callback=progress_callback,
    )
    if result is None:
        logger.error("Cell performance computation returned None")
        return {}
    # IMPORTANT: Do not exclude None values. The persisted / validated JSON schema
    # requires some nullable fields (e.g. BOM item `electrode`) to be present even
    # when their value is null.
    return result.model_dump(exclude_none=False)  # type: ignore[return-value]


def summarize_for_copilot(full_result: dict[str, Any]) -> dict[str, Any]:
    """Produce a compact, curated summary for copilot context.

    Explicit field selection: simulated KPIs (PyBaMM-derived, may be None if skipped),
    theoretical targets, cell sizing, voltage limits, stoichiometry. Drops internals
    like electrode porosity, density, volume fractions, and large payloads.
    """
    if not full_result:
        return {}

    # Explicit field list grouped by meaning
    keys_to_keep = [
        # Simulated performance (PyBaMM-derived, may be None if sim skipped)
        "cell_nominal_capacity_Ah",
        "cell_nominal_energy_Wh",
        "cell_nominal_gravimetric_energy_density_Wh_kg",
        "cell_nominal_volumetric_energy_density_Wh_L",
        "cell_nominal_max_discharge_power_300s_W",
        "cell_nominal_gravimetric_power_density_W_kg",
        "cell_nominal_volumetric_power_density_W_L",
        "cell_dcir_10s_mohm",
        # Theoretical design targets
        "cell_theoretical_capacity_Ah",
        "cell_theoretical_energy_Wh",
        "cell_theoretical_gravimetric_energy_density_Wh_kg",
        "cell_theoretical_volumetric_energy_density_Wh_L",
        # Cell sizing
        "cell_mass_g",
        "cell_volume_L",
        "cell_n_p_ratio",
        # Voltage limits
        "ocv100_V",
        "ocv0_V",
        # Electrode stoichiometry (4 compact scalars)
        "stoichiometry_windows",
    ]

    out: dict[str, Any] = {}

    # Add selected KPI fields (preserve zero values, only drop actual None)
    for k in keys_to_keep:
        if k in full_result and full_result[k] is not None:
            out[k] = full_result[k]

    # Include error and conditions (important context even for partial/failed results)
    if full_result.get("error") is not None:
        out["error"] = full_result["error"]
    if full_result.get("conditions") is not None:
        out["conditions"] = full_result["conditions"]

    return out
