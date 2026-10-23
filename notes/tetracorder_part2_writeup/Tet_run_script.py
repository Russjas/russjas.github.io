"""
Run python-tetracorder over every Helmholtz benchmark sample, for every run setting in RUNS.

Each run writes one npz per sample into its own folder under Clean_runs, plus a run_settings.json recording exactly
how it was made, so a run's settings can be checked from the files rather than reconstructed.
"""
#%% Imports, paths and run settings
import json
from pathlib import Path

import hycore
import numpy as np
import scipy as sc
from tetracorderp import GroupEvaluator

BENCHMARK = Path("C:/Users/Hyperspectral/Documents/HS_Data/Helmholz_Benchmark_set")
shed_path = BENCHMARK / "MLA" / "MLA.shed"
CLEAN_DIR = BENCHMARK / "Tet_runs" / "Clean_runs"

SLICE = slice(10, -10, None)                                      # drop the noisy first and last 10 FENIX bands
SWIR_START = 970                                                  # nm: the SWIR detector starts at 974 nm

# run name -> Savitzky-Golay window and order, enabled groups, and whether only the SWIR detector is used
RUNS = {
    "run1_sg11-2_g1g2": {"window": 11, "poly": 2, "groups": {0, 1, 2}, "swir_only": False},
    "run2_sg11-2_g2": {"window": 11, "poly": 2, "groups": {0, 2}, "swir_only": False},
    "run3_sg5-1_g2": {"window": 5, "poly": 1, "groups": {0, 2}, "swir_only": False},
    "run4_sg11-2_g2_trueswir": {"window": 11, "poly": 2, "groups": {0, 2}, "swir_only": True},
}

#%% Rules and disabled materials
with open("C:/Users/Hyperspectral/Documents/GitHub/tetraCOREder/core_lookups.json") as f:
    Core_lookups = json.load(f)["Core_lookups"]

with open("C:/Users/Hyperspectral/Documents/GitHub/python-tetracorder/tetracorderp/resources/"
          "tetracorder_rules_as_dict.json") as f:
    tet_rules = json.load(f)["rules"]

disabled_materials = {k for k, v in Core_lookups.items() if v == "non-geo"} | {
    "chalcedony", "hydrated_basaltic_glass",                                   # broad single-feature, no NOTs
    "azurite.ws316.g2", "malachite.hs254.3b.g2", "nontronite.ng-1.a.g2",       # judged without their VNIR features
}
all_groups = {int(r["number"]) for r in tet_rules.values() if r["kind"] == "group"}
disabled_cases = {int(r["number"]) for r in tet_rules.values() if r["kind"] == "case"} - {6}

# one folder per run, with its settings written alongside the npz files
for run_name, settings in RUNS.items():
    (CLEAN_DIR / run_name).mkdir(parents=True, exist_ok=True)
    with open(CLEAN_DIR / run_name / "run_settings.json", "w") as f:
        json.dump({"savgol_window": settings["window"], "savgol_poly": settings["poly"],
                   "enabled_groups": sorted(settings["groups"]), "enabled_cases": [6],
                   "valid_bands_from_nm": SWIR_START if settings["swir_only"] else None,
                   "band_slice": [SLICE.start, SLICE.stop], "mode": "default", "temperature_K": [288, 298],
                   "disabled_materials": sorted(disabled_materials)}, f, indent=2)

#%% Run every sample with every setting
shed = hycore.loadShed(shed_path)
boxes = [b for b in shed.getBoxes() if "test" not in b.name.lower() and "train" not in b.name.lower()]

for box in boxes:
    identifier = f"{box.holeName()}_{box.name}"
    todo = {name: s for name, s in RUNS.items()
            if not (CLEAN_DIR / name / f"{identifier}_tetracorder.npz").exists()}
    if not todo:
        print(f"{identifier} done")
        continue

    fenix = box.get("FENIX")
    data = fenix.data[..., SLICE]
    wavelengths = fenix.get_wavelengths()[SLICE]
    fwhm = np.asarray(fenix.header.get_list("fwhm"), dtype=float)[SLICE]
    assert data.shape[-1] == len(wavelengths) == len(fwhm), (
        f"Shape mismatch: data={data.shape[-1]}, wavelengths={len(wavelengths)}, fwhm={len(fwhm)}")

    # Identify spectra with all finite bands - savgol chokes on NaNs
    valid_pixels = np.all(np.isfinite(data), axis=-1)

    for run_name, settings in todo.items():
        print(f"running {identifier}: {run_name}")

        # full range: every band used and smoothed; SWIR only: bands below 970 nm ignored and left unsmoothed
        valid_bands = wavelengths >= SWIR_START if settings["swir_only"] else np.ones(len(wavelengths), dtype=bool)

        run_cube = np.full(data.shape, np.nan, dtype=np.float32)
        run_cube[valid_pixels] = data[valid_pixels]               # VNIR left raw in SWIR-only runs: ignored anyway

        # Smooth only valid spectra, and only the valid bands
        smoothed = run_cube[valid_pixels]
        smoothed[:, valid_bands] = sc.signal.savgol_filter(smoothed[:, valid_bands], settings["window"],
                                                           settings["poly"], axis=-1)
        run_cube[valid_pixels] = smoothed

        run = GroupEvaluator(
            run_cube,                                  # The smoothed, pre-processed sample data
            wavelengths / 1000,                        # convert nm to um required by Tetracorder
            fwhm / 1000,                               # convert nm to um required by Tetracorder
            target_valid_bands=valid_bands,            # all bands, or the SWIR detector only
            mode="default",                            # Tetracorders default preset thresholds
            masked_pixels=~valid_pixels,               # mask pixels with NaN
            temperature=(288, 298),                    # Room temperature in kelvin
            disabled_materials=disabled_materials,     # Core scanning specific disabled list (non-geo and named)
            disabled_groups=all_groups - settings["groups"],
            disabled_cases=disabled_cases,
        )
        run.write_npz(CLEAN_DIR / run_name / f"{identifier}_tetracorder.npz")  # persist results for evaluation