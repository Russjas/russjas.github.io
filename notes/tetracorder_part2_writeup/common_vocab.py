"""
Common vocabulary tests over the full Helmholtz benchmark, for every Tetracorder run in RUNS.

The write-up runs each test on one sample (Geyer bG16). This script runs the same tests on every sample, loading each
sample's MLA once and comparing it with each run:

H1   Is Group 2's winner the most abundant SWIR-active mineral? Match levels, and precision, recall and MCC.
H2a  Does Tetracorder detect SWIR-active minerals at low abundance in a spectrally neutral matrix? Detection rates.
H2b  Is the mineral Tetracorder names actually there? The share of calls where the named group is present.
H3   Does Tetracorder's fit (or depth, or fit x depth) track the abundance of the mineral it names?

Outputs are CSVs in OUT_DIR: one set per run, plus tables comparing the runs.
"""
#%% Paths and settings
from pathlib import Path

import hycore
import numpy as np
import pandas as pd
from spectral.io import envi
from tetracorderp import GroupEvaluator

BENCHMARK = Path(r"C:/Users/Hyperspectral/Documents/HS_Data/Helmholz_Benchmark_set")
WRITEUP = Path(r"C:/Users/Hyperspectral/Documents/GitHub/russjas.github.io/notes/tetracorder_part2_writeup")
shed_path = BENCHMARK / "MLA" / "MLA.shed"
DOCS = BENCHMARK / "docs" 
OUT_DIR = DOCS / "results" / "common_vocab"

# run name -> (run folder, npz suffix); runs whose folder doesn't exist are skipped
RUNS = {
    "run1_sg11-2_g1g2": (BENCHMARK / "Tet_runs" / "11-2_GROUP1on", "_tetracorder_run_11_2.npz"),
    "run2_sg11-2_g2": (BENCHMARK / "Tet_runs" / "11-2_NOGROUP1", "_tetracorder_run_11_2.npz"),
    "run3_sg5-1_g2": (BENCHMARK / "Tet_runs" / "5-1_NOGROUP1", "_tetracorder_run_5_1.npz"),
    "run4_sg11-2_g2_trueswir": (BENCHMARK / "Tet_runs" / "11-2_TRUSWIR_NOGROUP1", "_tetracorder_run_11_2.npz"),
}

SWIR_ACTIVE = {"Al-OH", "Fe/Mg-OH", "CO3", "H2O"}                # spectral groups with a SWIR absorption
VOCAB = ["species", "group", "spectral_group"]                   # the three levels of the common vocabulary
EXCLUDE = ["Other", "Unknown", "Porosity"]
MIN_DOMINANT = 0.10                                               # H1: dominant SWIR-active mineral at least 10%
MATCH_ORDER = ["species", "group", "spectral", "miss", "no call", "not in lookup"]
BINS = [-0.001, 0, 0.01, 0.025, 0.05, 0.10, 0.25, 1.0]            # H2a and H3 abundance bins
LABELS = ["absent", "0-1%", "1-2.5%", "2.5-5%", "5-10%", "10-25%", ">25%"]
NEUTRAL_MAX = 0.01                                                # H2a: other SWIR-active minerals under 1%

OUT_DIR.mkdir(parents=True, exist_ok=True)

#%% Lookups
rule_lookup = pd.read_csv(WRITEUP / "group2_crosswalk_long.csv", keep_default_na=False)
mla_lookup = pd.read_csv(WRITEUP / "mla_channels.csv", keep_default_na=False).set_index("mla_channel")

# everything each rule names, at every level (a mixture names several), built once rather than per pixel
rule_sets = {rule: {level: set(tet[level]) - {""} for level in VOCAB} for rule, tet in rule_lookup.groupby("rule_id")}


# the finest level at which a winner and an MLA channel agree
def match_level(tet_sets, truth):
    for level, name in zip(VOCAB, ["species", "group", "spectral"]):
        if truth[level] != "" and truth[level] in tet_sets[level]:
            return name
    return "miss"


#%% Compare every sample with every run
shed = hycore.loadShed(shed_path)
boxes = [b for b in shed.getBoxes() if "test" not in b.name.lower() and "train" not in b.name.lower()]

runs = {name: (run_dir, suffix) for name, (run_dir, suffix) in RUNS.items() if run_dir.exists()}
for name in RUNS.keys() - runs.keys():
    print(f"{name}: {RUNS[name][0]} not found, skipped")

tables = {name: {"h1": [], "h2a": [], "h2b": [], "h3": []} for name in runs}
reported_groups = {}                                              # sample -> mineral groups its MLA reports
match_cache = {}                                                  # (rule, MLA channel) -> match level

for box in boxes:                                                 # iterate, don't getBox: Freiberg names lack the "b"
    sample_id = f"{box.holeName()}_{box.name}"
    mla_path = shed_path / f"{box.holeName()}.hyc" / f"{box.name}.hyc" / "MLA_HSI.hdr"
    if not mla_path.exists():
        print(f"{sample_id}: no MLA, skipped")
        continue

    mla_img = envi.open(str(mla_path))
    mineral_channels = [n.strip() for n in mla_img.metadata["band names"]]
    mla = np.asarray(mla_img.load(), dtype=np.float32).transpose(1, 0, 2)  # ENVI -> hylite (x, y)
    mla = mla[..., :len(mineral_channels)]                        # some samples carry an empty, unnamed last band

    active_mask = mla_lookup.loc[mineral_channels, "spectral_group"].isin(SWIR_ACTIVE).to_numpy()
    exclude_mask = np.isin(mineral_channels, EXCLUDE)
    channel_group = mla_lookup.loc[mineral_channels, "group"].to_numpy()
    reported = {channel_group[i] for i in np.flatnonzero(active_mask) if np.isfinite(mla[..., i]).any()}
    reported_groups[sample_id] = reported

    # MLA side, every pixel at once: skip no data, or more than 70% Porosity, Unknown, Other
    abundances = np.nan_to_num(mla)
    valid = (abundances.sum(axis=-1) > 0) & (abundances[..., exclude_mask].sum(axis=-1) <= 0.7)
    xs, ys = np.nonzero(valid)
    active = np.where(active_mask, abundances, 0)[xs, ys]        # (pixels, channels), SWIR-active channels only

    # the dominant mineral: the most abundant SWIR-active channel, if it makes up at least 10%
    dominant_index = active.argmax(axis=1)
    dominant_abundance = active.max(axis=1)
    has_dominant = dominant_abundance >= MIN_DOMINANT
    dominant = np.where(has_dominant, np.array(mineral_channels, dtype=object)[dominant_index], "")
    dominant_group = np.where(has_dominant, channel_group[dominant_index], "")

    # the abundance of each SWIR-active mineral group, and whether the rest of the pixel is a neutral matrix
    groups = sorted(set(channel_group[active_mask]))
    group_index = {g: i for i, g in enumerate(groups)}
    by_group = np.zeros((len(xs), len(groups)))
    for i, g in enumerate(groups):
        by_group[:, i] = active[:, channel_group == g].sum(axis=1)
    neutral = (by_group.sum(axis=1, keepdims=True) - by_group) < NEUTRAL_MAX

    for run_name, (run_dir, suffix) in runs.items():
        run_path = run_dir / f"{sample_id}{suffix}"
        if not run_path.exists():
            print(f"{run_name} {sample_id}: missing run")
            continue
        result = GroupEvaluator.from_npz(run_path).group_winners[2]
        if result["winner"].shape != mla.shape[:2]:
            print(f"{run_name} {sample_id}: run {result['winner'].shape} vs MLA {mla.shape[:2]}, skipped")
            continue

        # Tetracorder side: the winner, its fit, depth and fit x depth, and the mineral groups it names
        winners = np.array([rule or "" for rule in result["winner"][xs, ys]], dtype=object)
        fit, depth, fit_depth = (result[k][xs, ys] for k in ("fit", "depth", "fit_depth"))
        called = [rule_sets[rule]["group"] if rule in rule_sets else set() for rule in winners]

        # H1: the finest level at which the winner agrees with the dominant mineral
        match = []
        for rule, channel in zip(winners, dominant):
            if not channel:
                match.append("")
            elif not rule:
                match.append("no call")
            elif rule not in rule_sets:
                match.append("not in lookup")                     # a winner the crosswalk doesn't know: fix the lookup
            else:
                if (rule, channel) not in match_cache:
                    match_cache[rule, channel] = match_level(rule_sets[rule], mla_lookup.loc[channel, VOCAB])
                match.append(match_cache[rule, channel])
        tables[run_name]["h1"].append(pd.DataFrame({
            "sample": sample_id, "hole": box.holeName(), "x": xs, "y": ys, "winner": winners,
            "dominant": dominant, "dominant_group": dominant_group, "dominant_abundance": dominant_abundance,
            "called_groups": [";".join(sorted(c)) for c in called], "match": match}))

        # H2a: each reported mineral group in a neutral matrix: how much is there, and did Tetracorder call it?
        is_called = np.array([[g in c for g in groups] for c in called], dtype=bool).reshape(len(xs), -1)
        pixel, column = np.nonzero(neutral & np.isin(groups, list(reported)))
        tables[run_name]["h2a"].append(pd.DataFrame({
            "sample": sample_id, "group": np.array(groups, dtype=object)[column],
            "abundance": by_group[pixel, column], "called": is_called[pixel, column]}))

        # H2b and H3: every call, and the abundance of each mineral group it names
        h2b, h3 = [], []
        for p, named in enumerate(called):
            if not named:
                continue                                          # no call, or a winner the crosswalk doesn't know
            for g in named:
                abundance = by_group[p, group_index[g]] if g in group_index else 0.0
                h2b.append((sample_id, winners[p], g, len(named) > 1, g in reported, abundance))
                if len(named) == 1:
                    h3.append((sample_id, winners[p], g, abundance, fit[p], depth[p], fit_depth[p]))
        tables[run_name]["h2b"].append(pd.DataFrame(
            h2b, columns=["sample", "winner", "group", "mixture", "reported", "abundance"]))
        tables[run_name]["h3"].append(pd.DataFrame(
            h3, columns=["sample", "winner", "group", "abundance", "fit", "depth", "fit_depth"]))

tables = {name: {k: pd.concat(v, ignore_index=True) for k, v in t.items()} for name, t in tables.items() if t["h1"]}
for run_name, t in tables.items():
    t["h1"].to_csv(OUT_DIR / f"{run_name}_h1_pixels.csv", index=False)
    print(f"{run_name}: {len(t['h1']):,} pixels from {t['h1']['sample'].nunique()} samples")
    unknown = t["h1"].loc[t["h1"]["match"] == "not in lookup", "winner"].value_counts()
    if not unknown.empty:
        print(f"{run_name}: winners missing from the crosswalk:\n{unknown.to_string()}")

#%% H1: match levels, for pixels with a dominant SWIR-active mineral
for run_name, t in tables.items():
    scored = t["h1"][t["h1"]["dominant"] != ""]
    match_levels = pd.crosstab(scored["dominant_group"], scored["match"], normalize="index")
    match_levels = match_levels.reindex(columns=MATCH_ORDER, fill_value=0)
    match_levels.insert(0, "n_pixels", scored["dominant_group"].value_counts())
    match_levels = match_levels.sort_values("n_pixels", ascending=False)
    match_levels.to_csv(OUT_DIR / f"{run_name}_h1_match_levels.csv")
    print(f"\n{run_name}: H1 match levels by dominant SWIR-active group (fraction of pixels)")
    print(match_levels.round(2).to_string())

#%% H1: precision, recall and MCC, yes/no per mineral group over every pixel
h1_scores = []
for run_name, t in tables.items():
    pixels = t["h1"]
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
    scores.to_csv(OUT_DIR / f"{run_name}_h1_scores.csv")
    h1_scores.append(scores.reset_index())
    print(f"\n{run_name}: H1 precision, recall and MCC per mineral group")
    print(scores.drop(columns="run").round(2).to_string())
h1_scores = pd.concat(h1_scores, ignore_index=True)
h1_scores.to_csv(OUT_DIR / "h1_scores_all_runs.csv", index=False)

#%% H2a: detection rate against abundance, in a spectrally neutral matrix
h2a_long = []
for run_name, t in tables.items():
    detection = t["h2a"].assign(bin=pd.cut(t["h2a"]["abundance"], BINS, labels=LABELS))
    present = detection.groupby("group")["bin"].apply(lambda b: (b != "absent").any())
    detection = detection[detection["group"].isin(present[present].index)]  # present in some neutral pixel

    # of the neutral pixels with a mineral group at a given abundance, how many did Tetracorder call it?
    # The "absent" column is the false call rate.
    rates = pd.crosstab(detection["group"], detection["bin"], values=detection["called"], aggfunc="mean")
    counts = pd.crosstab(detection["group"], detection["bin"])           # pixel counts behind each rate
    rates.to_csv(OUT_DIR / f"{run_name}_h2a_detection_rates.csv")
    counts.to_csv(OUT_DIR / f"{run_name}_h2a_detection_counts.csv")
    print(f"\n{run_name}: H2a detection rate by abundance, spectrally neutral matrix")
    print(rates.round(2).to_string())
    print(f"\n{run_name}: H2a pixel counts")
    print(counts.to_string())

    long = detection.groupby(["group", "bin"], observed=True)["called"].agg(n_pixels="size", detection_rate="mean")
    h2a_long.append(long.reset_index().assign(run=run_name))
h2a_long = pd.concat(h2a_long, ignore_index=True)
h2a_long.to_csv(OUT_DIR / "h2a_detection_all_runs.csv", index=False)

#%% H2b: is the mineral Tetracorder names actually there?
h2b_scores = []
for run_name, t in tables.items():
    named_calls = t["h2b"]

    # calls the MLA can judge: how often is the named mineral group there at all?
    judged = named_calls[named_calls["reported"]]
    summary = judged.groupby("group")["abundance"].agg(
        n_calls="size",
        present=lambda a: (a > 0).mean(),                           # any amount
        present_1pct=lambda a: (a >= 0.01).mean(),                  # 1% or more
        median_abundance=lambda a: a[a > 0].median())               # how much, when it's there
    summary = summary.sort_values("n_calls", ascending=False)
    summary.to_csv(OUT_DIR / f"{run_name}_h2b_named_present.csv")
    print(f"\n{run_name}: H2b share of calls where the named mineral group is present")
    print(summary.round(3).to_string())

    # calls the MLA can't judge: the named group isn't measured in that sample
    not_reported = named_calls.loc[~named_calls["reported"], "group"].value_counts()
    not_reported.to_csv(OUT_DIR / f"{run_name}_h2b_not_reported.csv")
    print(f"\n{run_name}: H2b calls the MLA can't judge")
    print(not_reported.to_string())
    h2b_scores.append(summary.reset_index().assign(run=run_name))
h2b_scores = pd.concat(h2b_scores, ignore_index=True)
h2b_scores.to_csv(OUT_DIR / "h2b_named_present_all_runs.csv", index=False)

#%% H3: does fit, depth or fit x depth track the abundance of the mineral Tetracorder names?
h3_rho = []
for run_name, t in tables.items():
    # correct calls only: the winner names a single mineral group, and the MLA says that group is there
    correct = t["h3"][t["h3"]["abundance"] > 0]

    # Spearman correlation with abundance: 1 = rises with abundance, 0 = unrelated
    rho = correct.groupby("group")[["abundance", "fit", "depth", "fit_depth"]].corr(method="spearman")
    rho = rho.xs("abundance", level=1)[["fit", "depth", "fit_depth"]]
    rho.insert(0, "n_pixels", correct["group"].value_counts())
    rho = rho.sort_values("n_pixels", ascending=False)
    rho.to_csv(OUT_DIR / f"{run_name}_h3_spearman.csv")
    print(f"\n{run_name}: H3 Spearman correlation with MLA abundance, correct single-group calls")
    print(rho.round(2).to_string())

    # median fit, depth and fit x depth in each abundance bin
    correct = correct.assign(bin=pd.cut(correct["abundance"], BINS, labels=LABELS))
    for metric in ["fit", "depth", "fit_depth"]:
        medians = correct.pivot_table(index="group", columns="bin", values=metric, aggfunc="median", observed=True)
        medians.to_csv(OUT_DIR / f"{run_name}_h3_median_{metric}.csv")
        if metric == "depth":
            print(f"\n{run_name}: H3 median depth by abundance bin")
            print(medians.round(3).to_string())
    h3_rho.append(rho.reset_index().assign(run=run_name))
h3_rho = pd.concat(h3_rho, ignore_index=True)
h3_rho.to_csv(OUT_DIR / "h3_spearman_all_runs.csv", index=False)

#%% Runs side by side
print("\nH1 MCC by run")
print(h1_scores.pivot(index="group", columns="run", values="mcc").round(2).to_string())

print("\nH2a detection rate at 1-2.5% by run")
low = h2a_long[h2a_long["bin"] == "1-2.5%"]
print(low.pivot(index="group", columns="run", values="detection_rate").round(2).to_string())

print("\nH2b share of calls with the named group present (any amount) by run")
print(h2b_scores.pivot(index="group", columns="run", values="present").round(2).to_string())

print("\nH3 Spearman correlation of depth with abundance by run")
print(h3_rho.pivot(index="group", columns="run", values="depth").round(2).to_string())
