"""
Common vocabulary tests: is Group 2's winner the most abundant SWIR-active mineral?

For each pixel: the most abundant SWIR-active MLA channel (the dominant mineral) and Tetracorder's Group 2 winner,
scored at the finest level they agree (species, group, spectral, miss), or no call. Then, for each mineral group, a
yes/no question on every pixel: did Tetracorder call it, and is it the dominant SWIR-active group? Precision, recall
and MCC come from that, pooled over all samples whose MLA reports the group.

Inputs: the MLA shed, the Tetracorder runs (npz), group2_crosswalk_long.csv and mla_channels.csv.
Outputs, per run, in OUT_DIR: <run>_pixels.csv, <run>_match_levels.csv, <run>_group_scores.csv; plus
group_scores_all_runs.csv.
"""
#%% Paths and settings
from pathlib import Path

import hycore
import numpy as np
import pandas as pd
from spectral.io import envi
from tetracorderp import GroupEvaluator

BENCHMARK = Path(r"C:/Users/Hyperspectral/Documents/HS_Data/Helmholz_Benchmark_set")
shed_path = BENCHMARK / "MLA" / "MLA.shed"
DOCS = BENCHMARK / "docs"
OUT_DIR = DOCS / "results"

# run name -> (run folder, npz suffix); runs whose folder doesn't exist are skipped
RUNS = {
    "run1_sg11-2_g1g2": (BENCHMARK / "Tet_runs" / "11-2_Group1", "_tetracorder_run_11_2.npz"),
    "run2_sg11-2_g2": (BENCHMARK / "Tet_runs" / "11-2_NOGROUP1", "_tetracorder_run_11_2.npz"),
    "run3_sg5-1_g2": (BENCHMARK / "Tet_runs" / "5-1_NOGROUP1", "_tetracorder_run_5_1.npz"),
}

SWIR_ACTIVE = {"Al-OH", "Fe/Mg-OH", "CO3", "H2O"}
VOCAB = ["species", "group", "spectral_group"]
MIN_DOMINANT = 0.10                                               # dominant SWIR-active mineral must be at least 10%
EXCLUDE = ["Other", "Unknown", "Porosity"]
MATCH_ORDER = ["species", "group", "spectral", "miss", "no call", "not in lookup"]

OUT_DIR.mkdir(exist_ok=True)

#%% Lookups
rule_lookup = pd.read_csv(DOCS / "group2_crosswalk_long.csv", keep_default_na=False)
mla_lookup = pd.read_csv(DOCS / "mla_channels.csv", keep_default_na=False).set_index("mla_channel")

# every species/group/spectral group each rule names (a mixture names several), built once rather than per pixel
rule_sets = {rule: {level: set(tet[level]) - {""} for level in VOCAB} for rule, tet in rule_lookup.groupby("rule_id")}


# the finest level at which they agree
def match_level(tet_sets, truth):
    for level, name in zip(VOCAB, ["species", "group", "spectral"]):
        if truth[level] != "" and truth[level] in tet_sets[level]:
            return name
    return "miss"


#%% Pixel tables: every sample, every run
shed = hycore.loadShed(shed_path)
boxes = shed.getBoxes()

pixel_tables = {}
reported_groups = {}                                              # sample -> mineral groups its MLA reports
for run_name, (run_dir, suffix) in RUNS.items():
    if not run_dir.exists():
        print(f"{run_name}: {run_dir} not found, skipped")
        continue

    rows = []
    for box in boxes:                                             # iterate, don't getBox: Freiberg names lack the "b"
        sample_id = f"{box.holeName()}_{box.name}"
        run_path = run_dir / f"{sample_id}{suffix}"
        mla_path = shed_path / f"{box.holeName()}.hyc" / f"{box.name}.hyc" / "MLA_HSI.hdr"
        if not run_path.exists() or not mla_path.exists():
            print(f"{run_name} {sample_id}: missing {'run' if not run_path.exists() else 'MLA'}")
            continue

        run = GroupEvaluator.from_npz(run_path)
        winners = run.group_winners[2]["winner"]
        fit = run.group_winners[2]["fit"]
        depth = run.group_winners[2]["depth"]

        mla_img = envi.open(str(mla_path))
        mineral_channels = [n.strip() for n in mla_img.metadata["band names"]]
        mla = np.asarray(mla_img.load(), dtype=np.float32).transpose(1, 0, 2)  # ENVI -> hylite (x, y)
        mla = mla[..., :len(mineral_channels)]                    # some samples carry an empty, unnamed last band
        if winners.shape != mla.shape[:2]:
            print(f"{run_name} {sample_id}: run {winners.shape} vs MLA {mla.shape[:2]}, skipped")
            continue

        active_mask = mla_lookup.loc[mineral_channels, "spectral_group"].isin(SWIR_ACTIVE).to_numpy()
        exclude_mask = np.isin(mineral_channels, EXCLUDE)
        channel_group = mla_lookup.loc[mineral_channels, "group"].to_numpy()
        reported_groups[sample_id] = {channel_group[i] for i in np.flatnonzero(active_mask)
                                      if np.isfinite(mla[..., i]).any()}

        for x, y in np.ndindex(winners.shape):
            abundances = np.nan_to_num(mla[x, y])
            if abundances.sum() == 0 or abundances[exclude_mask].sum() > 0.7:
                continue                                          # no MLA data, or >70% Porosity, Unknown, Other

            # MLA side: the most abundant SWIR-active channel, and its mineral group
            active = np.where(active_mask, abundances, 0)
            dominant = mineral_channels[active.argmax()] if active.max() >= MIN_DOMINANT else ""

            # Tetracorder side: the winner and everything it names (a mixture names several)
            rule = winners[x, y]
            tet_sets = rule_sets.get(rule, {level: set() for level in VOCAB})

            # match level: the finest level at which the winner agrees with the dominant mineral
            if not dominant:
                match = ""
            elif not rule:
                match = "no call"
            elif rule not in rule_sets:
                match = "not in lookup"                           # a winner the crosswalk doesn't know: fix the lookup
            else:
                match = match_level(tet_sets, mla_lookup.loc[dominant, VOCAB])

            rows.append({"sample": sample_id, "hole": box.holeName(), "x": x, "y": y,
                         "winner": rule or "", "fit": fit[x, y], "depth": depth[x, y],
                         "dominant": dominant,
                         "dominant_group": channel_group[active.argmax()] if dominant else "",
                         "dominant_abundance": active.max(),
                         "called_groups": ";".join(sorted(tet_sets["group"])), "match": match})

    pixels = pd.DataFrame(rows)
    pixels.to_csv(OUT_DIR / f"{run_name}_pixels.csv", index=False)
    pixel_tables[run_name] = pixels
    print(f"{run_name}: {len(pixels):,} pixels from {pixels['sample'].nunique()} samples")
    unknown = pixels.loc[pixels["match"] == "not in lookup", "winner"].value_counts()
    if not unknown.empty:
        print(f"{run_name}: winners missing from the crosswalk:\n{unknown.to_string()}")

#%% Match levels: for pixels with a dominant SWIR-active mineral, how does the winner compare?
for run_name, pixels in pixel_tables.items():
    scored = pixels[pixels["dominant"] != ""]
    match_levels = pd.crosstab(scored["dominant_group"], scored["match"], normalize="index")
    match_levels = match_levels.reindex(columns=MATCH_ORDER, fill_value=0)
    match_levels.insert(0, "n_pixels", scored["dominant_group"].value_counts())
    match_levels = match_levels.sort_values("n_pixels", ascending=False)
    match_levels.to_csv(OUT_DIR / f"{run_name}_match_levels.csv")
    print(f"\n{run_name}: match levels by dominant SWIR-active group (fraction of pixels)")
    print(match_levels.round(2).to_string())

#%% Precision, recall and MCC: yes/no per mineral group, over every pixel
all_scores = []
for run_name, pixels in pixel_tables.items():
    called_groups = pixels["called_groups"].str.split(";").apply(set)
    rows = []
    for g in sorted(set().union(*reported_groups.values())):
        in_scope = pixels["sample"].map(lambda s: g in reported_groups[s])  # only samples whose MLA reports g
        called = called_groups[in_scope].apply(lambda s: g in s)
        truth = pixels.loc[in_scope, "dominant_group"] == g
        tp, fp = (called & truth).sum(), (called & ~truth).sum()
        fn, tn = (~called & truth).sum(), (~called & ~truth).sum()
        if tp + fp + fn == 0:
            continue                                              # never dominant and never called: nothing to score
        denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        rows.append({"run": run_name, "group": g, "n_samples": pixels.loc[in_scope, "sample"].nunique(),
                     "n_dominant": tp + fn, "n_called": tp + fp,
                     "precision": tp / (tp + fp) if tp + fp else np.nan,  # when Tetracorder calls g, how often right?
                     "recall": tp / (tp + fn) if tp + fn else np.nan,     # share of g-dominated pixels called g
                     "mcc": (tp * tn - fp * fn) / denom if denom else np.nan})  # -1 to 1, 0 = chance
    scores = pd.DataFrame(rows).set_index("group").sort_values("n_dominant", ascending=False)
    scores.to_csv(OUT_DIR / f"{run_name}_group_scores.csv")
    all_scores.append(scores.reset_index())
    print(f"\n{run_name}: precision, recall and MCC per mineral group")
    print(scores.drop(columns="run").round(2).to_string())

#%% Runs side by side: MCC per mineral group
if all_scores:
    all_scores = pd.concat(all_scores, ignore_index=True)
    all_scores.to_csv(OUT_DIR / "group_scores_all_runs.csv", index=False)
    print("\nMCC by run")
    print(all_scores.pivot(index="group", columns="run", values="mcc").round(2).to_string())
    
#%%Delete below before publish
#%% Do runs 1 and 2 differ in anything except Group 1?
r1 = GroupEvaluator.from_npz(RUNS["run1_sg11-2_g1g2"][0] / f"Geyer_bG16{RUNS['run1_sg11-2_g1g2'][1]}")
r2 = GroupEvaluator.from_npz(RUNS["run2_sg11-2_g2"][0] / f"Geyer_bG16{RUNS['run2_sg11-2_g2'][1]}")
print(r1.mode, r2.mode, r1.disabled_groups ^ r2.disabled_groups)
print("disabled in run 1 only:", sorted(r1.disabled_materials - r2.disabled_materials))
print("disabled in run 2 only:", sorted(r2.disabled_materials - r1.disabled_materials))
#%% Where runs 1 and 2 disagree: which winners swap?
keys = ["sample", "x", "y"]
p1 = pd.read_csv(OUT_DIR / "run1_sg11-2_g1g2_pixels.csv", keep_default_na=False)
p2 = pd.read_csv(OUT_DIR / "run2_sg11-2_g2_pixels.csv", keep_default_na=False)
both = p1.merge(p2, on=keys, suffixes=("_g1on", "_g1off"))
changed = both[both["winner_g1on"] != both["winner_g1off"]]
print(f"{len(changed):,} of {len(both):,} pixels change winner ({len(changed) / len(both):.1%})")
print(changed.groupby(["winner_g1off", "winner_g1on"]).size().sort_values(ascending=False).head(20).to_string())
#%% Same inputs? And does a no-NOT rule's fit change?
print(r1.wavelengths.shape, r2.wavelengths.shape, r1.wavelengths.min(), r2.wavelengths.min())
print("valid bands equal:", r1.valid_bands.shape == r2.valid_bands.shape and (r1.valid_bands == r2.valid_bands).all())
same = r1.group_winners[2]["winner"] == r2.group_winners[2]["winner"]
d_fit = np.abs(r1.group_winners[2]["fit"] - r2.group_winners[2]["fit"])[same]
print(f"same winner on {same.mean():.0%} of pixels; max fit difference there: {np.nanmax(d_fit):.2e}")
