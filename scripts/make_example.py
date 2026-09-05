"""Regenerate examples/two_designs.json.

The base design mirrors the pouch ~80 Ah NMC811/graphite fixture used
by the upstream model's own tests, so the example is known-good rather
than invented. The second design varies one lever (positive mass
loading) — the shape a real sweep takes.

    python scripts/make_example.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "two_designs.json"


def base_cell_parameters() -> dict:
    return {
        "form_factor": "Pouch",
        "cell_width_mm": 220.0,
        "cell_height_mm": 140.0,
        "cell_thickness_mm": 12.0,
        "positive_electrode_mass_loading_mg_cm2": 15.0,
        "negative_electrode_mass_loading_mg_cm2": 12.0,
        "positive_electrode_sheet_count": 52.0,
        "negative_electrode_sheet_count": 53.0,
        "positive_coating_thickness_um": 70.0,
        "negative_coating_thickness_um": 80.0,
        "positive_electrode_specific_heat_capacity_J_kg_K": 700.0,
        "negative_electrode_specific_heat_capacity_J_kg_K": 700.0,
        "positive_electrode_thermal_conductivity_W_m_K": 1.0,
        "negative_electrode_thermal_conductivity_W_m_K": 1.0,
        "positive_electrode_electronic_conductivity_S_m": 100.0,
        "negative_electrode_electronic_conductivity_S_m": 100.0,
        "positive_electrode_active_materials": [
            {
                "name": "NMC811",
                "mass_fraction": 0.96,
                "density_g_cm3": 4.8,
                "specific_capacity_mAh_g": 200.0,
                "nominal_voltage_V": 3.77,
            }
        ],
        "negative_electrode_active_materials": [
            {
                "name": "Graphite",
                "mass_fraction": 0.98,
                "density_g_cm3": 2.26,
                "specific_capacity_mAh_g": 360.0,
                "nominal_voltage_V": 0.12,
            }
        ],
        "positive_electrode_binders": [
            {"name": "PVDF", "mass_fraction": 0.02, "density_g_cm3": 1.78}
        ],
        "negative_electrode_binders": [
            {"name": "CMC", "mass_fraction": 0.01, "density_g_cm3": 1.5},
            {"name": "SBR", "mass_fraction": 0.01, "density_g_cm3": 1.5},
        ],
        "positive_electrode_conductive_agents": [
            {"name": "Carbon Black", "mass_fraction": 0.02, "density_g_cm3": 1.8}
        ],
        "negative_electrode_conductive_agents": [],
        "jelly_roll_count": 1,
        "electrode_coating_side_count": 2,
        "positive_electrode_foil_thickness_um": 14.0,
        "negative_electrode_foil_thickness_um": 6.0,
        "positive_electrode_foil_density_g_cm3": 2.7,
        "negative_electrode_foil_density_g_cm3": 8.96,
        "positive_electrode_foil_electronic_conductivity_S_m": 3.5e7,
        "negative_electrode_foil_electronic_conductivity_S_m": 6.0e7,
        "positive_electrode_foil_specific_heat_capacity_J_kg_K": 900.0,
        "negative_electrode_foil_specific_heat_capacity_J_kg_K": 385.0,
        "positive_electrode_foil_thermal_conductivity_W_m_K": 237.0,
        "negative_electrode_foil_thermal_conductivity_W_m_K": 401.0,
        "separator_sheet_count": 106.0,
        "separator_sheet_width_mm": 220.0,
        "separator_sheet_height_mm": 140.0,
        "separator_thickness_um": 12.0,
        "separator_porosity": 0.42,
        "separator_density_g_cm3": 0.5,
        "separator_specific_heat_capacity_J_kg_K": 1000.0,
        "separator_thermal_conductivity_W_m_K": 0.334,
        "electrolyte_density_g_cm3": 1.2,
        "electrolyte_fill_ratio": 0.85,
        "casing_thickness_mm": 0.15,
        "casing_density_g_cm3": 2.7,
        "electrode_overhang_mm": 1.0,
        "jelly_roll_inner_diameter_mm": 4.0,
        "volume_packing_ratio": 0.95,
        "upper_voltage_cutoff_V": 4.2,
        "lower_voltage_cutoff_V": 3.0,
    }


def base_simulation_parameters() -> dict:
    return {
        "solver_atol": 1e-3,
        "solver_rtol": 1e-3,
        "calibration_rate_C": 0.05,
        "perform_rpt": False,
        "upper_voltage_cutoff_V": 4.2,
        "lower_voltage_cutoff_V": 3.0,
        "cell_contact_resistance_Ohm": None,
        "cell_cooling_surface_area_m2": 0.02,
        "cell_heat_transfer_coefficient_W_m2_K": 10.0,
        "temperature_safety_threshold_K": 363.15,
        "anode_potential_safety_threshold_V": 0.01,
        "rpt_dcir": None,
        "rpt_power": None,
        "first_cycle_coulombic_loss_pct": 5.0,
        "ambient_temperature_K": 298.15,
        "reference_cell_temperature_K": 298.15,
        "start_soc_pct": 100.0,
    }


def build_example() -> dict:
    thin = base_cell_parameters()
    thick = copy.deepcopy(thin)
    # One lever moved: heavier cathode loading, thicker coating to match.
    thick["positive_electrode_mass_loading_mg_cm2"] = 21.0
    thick["positive_coating_thickness_um"] = 98.0

    return {
        "designs": [
            {
                "id": "pouch-80ah-baseline",
                "name": "Baseline (15 mg/cm2 cathode)",
                "cell_parameters": thin,
            },
            {
                "id": "pouch-80ah-high-loading",
                "name": "High loading (21 mg/cm2 cathode)",
                "cell_parameters": thick,
            },
        ],
        "simulation_parameters": base_simulation_parameters(),
        "fail_fast": False,
        "max_workers": 1,
        "result_detail": "kpis",
    }


if __name__ == "__main__":
    EXAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLE_PATH.write_text(json.dumps(build_example(), indent=2) + "\n")
    print(f"Wrote {EXAMPLE_PATH}")
