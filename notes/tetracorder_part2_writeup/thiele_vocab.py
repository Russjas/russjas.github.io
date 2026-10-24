"""
Thiele et al. vocabulary tests over the benchmark's test sections, for every Tetracorder run in RUNS.

Tetracorder is scored the way the benchmark scores its trained models: the paper's 22 mineral groups, its 61 test
sections, its pixel mask (2 x 2 edge erosion, the 22 groups summing to at least 70%, renormalised), occurrence above
2.5%, and the dominant phase over all 22 groups, quartz and feldspar included. The sections' own mask bands are not
used, so the pixel count is a little above the paper's.

Outputs are CSVs in OUT_DIR: one table per run, with the published model scores alongside, plus a table comparing the
runs.
"""
#%% Paths and settings
from pathlib import Path

import hycore
import numpy as np
import pandas as pd
from skimage.morphology import binary_erosion                    # the same function the benchmark's MLAStack uses
from spectral.io import envi
from tetracorderp import GroupEvaluator

BENCHMARK = Path(r"C:/Users/Hyperspectral/Documents/HS_Data/Helmholz_Benchmark_set")
WRITEUP = Path(r"C:/Users/Hyperspectral/Documents/GitHub/russjas.github.io/notes/tetracorder_part2_writeup")
DOCS = BENCHMARK / "docs"                                         # test_train_split.csv and paper_metrics_long.csv
shed_path = BENCHMARK / "MLA" / "MLA.shed"
OUT_DIR = DOCS / "results" / "thiele_vocab"
CLEAN_DIR = BENCHMARK / "Tet_runs" / "Clean_runs"
# run name -> (run folder, npz suffix); runs whose folder doesn't exist are skipped
RUNS = {
    "run1_sg11-2_g1g2": (CLEAN_DIR / "run1_sg11-2_g1g2", "_tetracorder.npz"),
    "run2_sg11-2_g2": (CLEAN_DIR / "run2_sg11-2_g2", "_tetracorder.npz"),
    "run3_sg5-1_g2": (CLEAN_DIR / "run3_sg5-1_g2", "_tetracorder.npz"),
    "run4_sg11-2_g2_trueswir": (CLEAN_DIR / "run4_sg11-2_g2_trueswir", "_tetracorder.npz"),
}



CLIP = 2                                                          # edge erosion (MLAStack clip=2)
CLOSURE = 0.7                                                     # the 22 groups must add up to at least 70%
OCCURRENCE = 0.025                                                # a group is present above 2.5%
PAPER_PIXELS = 50146                                              # test pixels in the paper

OUT_DIR.mkdir(parents=True, exist_ok=True)

#%% Lookups, the benchmark's test/train split and its published scores
rule_lookup = pd.read_csv(WRITEUP / "group2_crosswalk_long.csv", keep_default_na=False)
mla_lookup = pd.read_csv(WRITEUP / "mla_channels.csv", keep_default_na=False).set_index("mla_channel")

PAPER22 = sorted(set(mla_lookup["paper_group"]) - {""})
rule_paper = {rule: set(groups) - {""} for rule, groups in rule_lookup.groupby("rule_id")["paper_group"]}

split = pd.read_csv(DOCS / "test_train_split.csv").set_index("identifier")["split"].to_dict()
paper = pd.read_csv(DOCS / "paper_metrics_long.csv")             # occurrence MCC per model and group ("MCC")
paper_best = paper.sort_values("MCC").groupby("group").tail(1).set_index("group")
paper_rmlp = paper[paper["Model"] == "RMLP"].set_index("group")["MCC"]

#%% Score every test section against every run
shed = hycore.loadShed(shed_path)
boxes = [b for b in shed.getBoxes() if split.get(f"{b.holeName()}_{b.name}") == "test"]
print(f"{len(boxes)} test sections (paper: 61)")

runs = {name: (run_dir, suffix) for name, (run_dir, suffix) in RUNS.items() if run_dir.exists()}
for name in RUNS.keys() - runs.keys():
    print(f"{name}: {RUNS[name][0]} not found, skipped")

pieces = {name: {"occurrence": [], "dominant": [], "called": []} for name in runs}
for box in boxes:
    sample_id = f"{box.holeName()}_{box.name}"
    mla_path = shed_path / f"{box.holeName()}.hyc" / f"{box.name}.hyc" / "MLA_HSI.hdr"
    if not mla_path.exists():
        print(f"{sample_id}: no MLA, skipped")
        continue

    mla_img = envi.open(str(mla_path))
    mineral_channels = [n.strip() for n in mla_img.metadata["band names"]]
    mla = np.asarray(mla_img.load(), dtype=np.float32).transpose(1, 0, 2)  # ENVI -> hylite (x, y)
    mla = mla[..., :len(mineral_channels)]                        # some samples carry an empty, unnamed last band
    channel_paper = mla_lookup.loc[mineral_channels, "paper_group"].to_numpy()

    # MLA side: the abundance of each of the 22 groups in every pixel, SWIR-active or not
    filled = np.nan_to_num(mla)
    paper_abundance = np.stack([filled[..., channel_paper == pg].sum(axis=-1) for pg in PAPER22], axis=-1)

    # their mask: erode the section edges, drop pixels where the 22 groups add up to under 70%, renormalise the rest
    inside = binary_erosion((filled > 0).any(axis=-1), np.ones((CLIP, CLIP)))
    total = paper_abundance.sum(axis=-1)
    keep = inside & (total >= CLOSURE)
    paper_abundance = paper_abundance[keep] / total[keep, None]

    # the two truths: every group over 2.5% (occurrence), and the single most abundant of all 22 (dominant phase)
    occurrence = paper_abundance > OCCURRENCE
    dominant = paper_abundance.argmax(axis=1)

    for run_name, (run_dir, suffix) in runs.items():
        run_path = run_dir / f"{sample_id}{suffix}"
        if not run_path.exists():
            print(f"{run_name} {sample_id}: missing run")
            continue
        winners = GroupEvaluator.from_npz(run_path).group_winners[2]["winner"]
        if winners.shape != mla.shape[:2]:
            print(f"{run_name} {sample_id}: run {winners.shape} vs MLA {mla.shape[:2]}, skipped")
            continue

        # Tetracorder side: which of the 22 groups the winner names (a mixture names several; many rules name none)
        called = np.array([[pg in rule_paper.get(rule, set()) for pg in PAPER22] for rule in winners[keep]],
                          dtype=bool).reshape(-1, len(PAPER22))
        pieces[run_name]["occurrence"].append(occurrence)
        pieces[run_name]["dominant"].append(dominant)
        pieces[run_name]["called"].append(called)

#%% Scores per run, beside the published models
all_scores = []
for run_name, p in pieces.items():
    if not p["called"]:
        continue
    occurrence, dominant, called = (np.concatenate(p[k]) for k in ("occurrence", "dominant", "called"))
    print(f"\n{run_name}: {len(dominant):,} test pixels (paper: {PAPER_PIXELS:,})")

    # yes/no per group, against each truth: precision, recall and MCC as in H1
    rows = []
    for i, pg in enumerate(PAPER22):
        row = {"run": run_name, "group": pg}
        for name, truth in [("occ", occurrence[:, i]), ("dom", dominant == i)]:
            c = called[:, i]
            tp, fp = (c & truth).sum(), (c & ~truth).sum()
            fn, tn = (~c & truth).sum(), (~c & ~truth).sum()
            denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            row.update({f"{name}_n_true": tp + fn, f"{name}_n_called": tp + fp,
                        f"{name}_precision": tp / (tp + fp) if tp + fp else np.nan,
                        f"{name}_recall": tp / (tp + fn) if tp + fn else np.nan,
                        f"{name}_mcc": (tp * tn - fp * fn) / denom if denom else np.nan})
        row.update({"paper_best_model": paper_best["Model"].get(pg), "paper_best_occ_mcc": paper_best["MCC"].get(pg),
                    "paper_rmlp_occ_mcc": paper_rmlp.get(pg)})
        rows.append(row)

    scores = pd.DataFrame(rows).set_index("group").sort_values("paper_best_occ_mcc", ascending=False)
    scores.to_csv(OUT_DIR / f"{run_name}_thiele_scores.csv")
    all_scores.append(scores.reset_index())
    shown = scores[(scores["occ_n_true"] + scores["occ_n_called"] + scores["dom_n_true"]) > 0]  # skip empty groups
    print(shown[["occ_n_true", "occ_n_called", "occ_precision", "occ_recall", "occ_mcc", "dom_mcc",
                 "paper_best_model", "paper_best_occ_mcc", "paper_rmlp_occ_mcc"]].round(2).to_string())

#%% Runs side by side: occurrence MCC, with the published best model and RMLP
if all_scores:
    all_scores = pd.concat(all_scores, ignore_index=True)
    all_scores.to_csv(OUT_DIR / "thiele_scores_all_runs.csv", index=False)
    side = all_scores.pivot(index="group", columns="run", values="occ_mcc")
    side["paper_best"] = paper_best["MCC"]
    side["paper_rmlp"] = paper_rmlp
    side = side.sort_values("paper_best", ascending=False)
    side.to_csv(OUT_DIR / "thiele_occurrence_mcc_all_runs.csv")
    print("\nOccurrence MCC by run, with the published best model and RMLP")
    print(side.round(2).to_string())
