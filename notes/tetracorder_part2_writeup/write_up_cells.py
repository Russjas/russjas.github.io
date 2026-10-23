
#%% Antomy of a Tetracorder run
from tetracorderp import GroupEvaluator

# an existing run saved to npz can be re-materialised 
run = GroupEvaluator.from_npz("C:/Users/Hyperspectral/Documents/HS_Data/Helmholz_Benchmark_set/Tet_runs/11-2_GROUP1on/Geyer_bG16_tetracorder_run_11_2.npz")
print(run.group_winners.keys())
#dict_keys([2]) run.group_winners a dictionary of all of the data for the groups analysed
print(run.group_winners[2].keys())
#dict_keys(['winner', 'fit', 'depth', 'fit_depth'])
winners = run.group_winners[2]["winner"]
fit = run.group_winners[2]["fit"]
depth = run.group_winners[2]["depth"]
fit_depth = run.group_winners[2]["fit_depth"]

print(winners.shape, winners.dtype)
#(31, 45) object

print("Pixel winner: ", winners[15, 23])
print("Winner's fit: ",fit[15, 23])
#Pixel winner:  clinochlore.fe.sc-cca-1
#Winner's fit:  0.95764077
#%% Anatomy of an MLA dataset
from pathlib import Path
import numpy as np
import hycore
from spectral.io import envi

hole_name = "Geyer" 
sample_name = "bG16"

shed_path = Path("C:/Users/Hyperspectral/Documents/HS_Data/Helmholz_Benchmark_set/MLA/MLA.shed")

shed = hycore.loadShed(shed_path)
boxes = shed.getBoxes()
box = shed.getHole(hole_name).getBox(sample_name)
mla_path = (shed_path / f"{box.holeName()}.hyc"/ f"{box.name}.hyc" / "MLA_HSI.hdr")
mla_img = envi.open(str(mla_path))
mineral_channels = [n.strip() for n in mla_img.metadata["band names"]]
mla = np.asarray(mla_img.load(), dtype=np.float32).transpose(1, 0, 2)  # ENVI -> hylite (x, y)  
print("mla shape: ", mla.shape)
print("number of mineral channels: ", len(mineral_channels))
#mla shape:  (31, 45, 70)
#number of mineral channels:  70
np.set_printoptions(suppress=True)
print(mla[15, 23])
#[0.3669482  0.18056996 0.01277622 ... ...]
print("Most abundant mineral in pixel: ", mineral_channels[np.argmax(mla[15,23])])
#Most abundant mineral in pixel:  Quartz
PRESENT = 0.001                                                   # Arbitrary presence threshold
present_mins = {mineral: round(float(ab), 3) for mineral, ab in zip(mineral_channels, mla[15, 23]) if ab > PRESENT}
for mineral, abundance in present_mins.items():
  print(mineral, abundance)
#Quartz 0.398
#Albite 0.248
#K-feldspar 0.332
#Chamosite Chl 0.012
#Muscovite Mica 0.008
#Apatite 0.001
#%% Using the look ups
import pandas as pd

SWIR_ACTIVE = {"Al-OH", "Fe/Mg-OH", "CO3", "H2O"}                # spectral groups with a SWIR absorption
VOCAB = ["species", "group", "spectral_group"]                   # the three levels of the common vocabulary

rule_lookup = pd.read_csv("C:/Users/Hyperspectral/Documents/GitHub/russjas.github.io/notes/tetracorder_part2_writeup/group2_crosswalk_long.csv", keep_default_na=False)
mla_lookup = pd.read_csv("C:/Users/Hyperspectral/Documents/GitHub/russjas.github.io/notes/tetracorder_part2_writeup/mla_channels.csv", keep_default_na=False).set_index("mla_channel")

# Tetracorder side: the example pixel's winner, and everything it names (a mixture names several)
rule = winners[15, 23]
tet_sets = {level: set(rule_lookup.loc[rule_lookup["rule_id"] == rule, level]) - {""} for level in VOCAB}
print(rule, tet_sets)
#clinochlore.fe.sc-cca-1 {'species': {'clinochlore'}, 'group': {'chlorite'}, 'spectral_group': {'Fe/Mg-OH'}}

# MLA side: the example pixel's most abundant SWIR-active channel
pixel = pd.Series(mla[15, 23], index=mineral_channels).fillna(0)
active = pixel[mla_lookup.loc[pixel.index, "spectral_group"].isin(SWIR_ACTIVE)]
channel = active.idxmax()
truth = mla_lookup.loc[channel, VOCAB]
print(channel, round(active.max(), 3), truth.to_dict())
#Chamosite Chl 0.012 {'species': 'chamosite', 'group': 'chlorite', 'spectral_group': 'Fe/Mg-OH'}


# the finest level at which a winner and an MLA channel agree
def match_level(tet_sets, truth):
    for level, name in zip(VOCAB, ["species", "group", "spectral"]):
        if truth[level] != "" and truth[level] in tet_sets[level]:
            return name
    return "miss"


print(match_level(tet_sets, truth))
#group



#%% H1: Is Group 2's winner the most abundant SWIR-active mineral?
MIN_DOMINANT = 0.10                                               # dominant SWIR-active mineral must be at least 10%
EXCLUDE = ["Other", "Unknown", "Porosity"]

active_mask = mla_lookup.loc[mineral_channels, "spectral_group"].isin(SWIR_ACTIVE).to_numpy()
exclude_mask = np.isin(mineral_channels, EXCLUDE)
channel_group = mla_lookup.loc[mineral_channels, "group"].to_numpy()

rows = []
for x, y in np.ndindex(winners.shape):
    abundances = np.nan_to_num(mla[x, y])
    if abundances.sum() == 0 or abundances[exclude_mask].sum() > 0.7:
        continue                                        # no MLA data, or more than 70% Porosity, Unknown, Other

    # MLA side: the most abundant SWIR-active channel, and its mineral group
    active = np.where(active_mask, abundances, 0)
    dominant = mineral_channels[active.argmax()] if active.max() >= MIN_DOMINANT else ""

    # Tetracorder side: the winner and everything it names (a mixture names several)
    rule = winners[x, y]
    tet = rule_lookup[rule_lookup["rule_id"] == rule]
    tet_sets = {level: set(tet[level]) - {""} for level in VOCAB}

    # match level: the finest level at which the winner agrees with the dominant mineral
    if not dominant:
        match = ""
    elif not rule:
        match = "no call"
    else:
        match = match_level(tet_sets, mla_lookup.loc[dominant, VOCAB])

    rows.append({"winner": rule,
                 "dominant": dominant,
                 "dominant_group": channel_group[active.argmax()] if dominant else "",
                 "called_groups": tet_sets["group"], "match": match})

sample = pd.DataFrame(rows)

# match levels, for pixels with a dominant SWIR-active mineral
print(sample.loc[sample["dominant"] != "", "match"].value_counts())
#match
#no call     349
#spectral      2
#Name: count, dtype: int64

# yes/no per mineral group, over every pixel: did Tetracorder call it, and is it the dominant SWIR-active group?
reported = sorted({channel_group[i] for i in np.flatnonzero(active_mask) if np.isfinite(mla[..., i]).any()})
for g in [g for g in reported if (sample["dominant_group"] == g).any()
          or sample["called_groups"].apply(lambda s: g in s).any()]:   # only groups this sample's MLA reports
    called = sample["called_groups"].apply(lambda s: g in s)
    truth = sample["dominant_group"] == g
    tp, fp = (called & truth).sum(), (called & ~truth).sum()
    fn, tn = (~called & truth).sum(), (~called & ~truth).sum()
    precision = tp / (tp + fp) if tp + fp else np.nan            # when Tetracorder calls g, how often is it right?
    recall = tp / (tp + fn) if tp + fn else np.nan               # of the g-dominated pixels, how many did it call g?
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else np.nan       # -1 to 1, 0 = no better than chance
    print(f"{g:12s} n={tp + fn:4d}  precision={precision:.2f}  recall={recall:.2f}  MCC={mcc:.2f}")
#biotite      n= 349  precision=0.00  recall=0.00  MCC=-0.03
#chlorite     n=   0  precision=0.00  recall=nan  MCC=nan
#smectite     n=   0  precision=0.00  recall=nan  MCC=nan
#white_mica   n=   2  precision=0.00  recall=0.00  MCC=-0.02
#%% H2a: does Tetracorder detect SWIR-active minerals at low abundance in a spectrally neutral matrix?
BINS = [-0.001, 0, 0.01, 0.025, 0.05, 0.10, 0.25, 1.0]
LABELS = ["absent", "0-1%", "1-2.5%", "2.5-5%", "5-10%", "10-25%", ">25%"]
NEUTRAL_MAX = 0.01                                                # other SWIR-active minerals under 1%: a neutral matrix

rows = []
for x, y in np.ndindex(winners.shape):
    abundances = np.nan_to_num(mla[x, y])
    if abundances.sum() == 0 or abundances[exclude_mask].sum() > 0.7:
        continue                                        # no MLA data, or more than 70% Porosity, Unknown, Other

    # MLA side: the abundance of each SWIR-active mineral group in the pixel
    by_group = pd.Series(abundances[active_mask], index=channel_group[active_mask]).groupby(level=0).sum()

    # Tetracorder side: the mineral groups the winner names (a mixture names several)
    rule = winners[x, y]
    called_groups = set(rule_lookup.loc[rule_lookup["rule_id"] == rule, "group"]) - {""}

    # one row per mineral group: how much is there, is the rest of the pixel neutral, did Tetracorder call it?
    for group, abundance in by_group.items():
        rows.append({"group": group,
                     "abundance": abundance,
                     "neutral": by_group.sum() - abundance < NEUTRAL_MAX,
                     "called": group in called_groups})

detection = pd.DataFrame(rows)
detection["bin"] = pd.cut(detection["abundance"], BINS, labels=LABELS)

# only neutral-matrix pixels, and only mineral groups present in at least one of them
neutral = detection[detection["neutral"]]
present = neutral.groupby("group")["bin"].apply(lambda b: (b != "absent").any())
neutral = neutral[neutral["group"].isin(present[present].index)]

# detection rate: of the neutral pixels with a mineral group at a given abundance, how many did Tetracorder call it?
# The "absent" column is the false call rate.
print(pd.crosstab(neutral["group"], neutral["bin"], values=neutral["called"], aggfunc="mean").round(2))
#bin         absent  0-1%  1-2.5%  2.5-5%  5-10%  10-25%
#group
#biotite        0.0   0.0    0.00    0.00    0.0     0.0
#chlorite       0.0   0.3    0.74    0.57    NaN     NaN
#epidote        0.0   0.0     NaN     NaN    NaN     NaN
#smectite       0.0   0.0     NaN     NaN    NaN     NaN
#white_mica     0.0   0.1    0.45    0.88    1.0     NaN
print(pd.crosstab(neutral["group"], neutral["bin"]))             # pixel counts behind each rate
#bin         absent  0-1%  1-2.5%  2.5-5%  5-10%  10-25%
#group
#biotite        115    11       8      19     13       4
#chlorite        70    84      61       7      0       0
#epidote        115    12       0       0      0       0
#smectite       116    12       0       0      0       0
#white_mica      70    82      33       8      1       0
#%% H2b: is the mineral Tetracorder names actually there?
rows = []
for x, y in np.ndindex(winners.shape):
    rule = winners[x, y]
    if not rule:
        continue                                        # no call: nothing named
    abundances = np.nan_to_num(mla[x, y])
    if abundances.sum() == 0 or abundances[exclude_mask].sum() > 0.7:
        continue                                        # no MLA data, or more than 70% Porosity, Unknown, Other

    # MLA side: the abundance of each SWIR-active mineral group in the pixel
    by_group = pd.Series(abundances[active_mask], index=channel_group[active_mask]).groupby(level=0).sum()

    # Tetracorder side: every mineral group the winner names, each checked on its own
    named = set(rule_lookup.loc[rule_lookup["rule_id"] == rule, "group"]) - {""}
    for group in named:
        rows.append({"winner": rule, "group": group, "mixture": len(named) > 1,
                     "reported": group in reported,                  # does this sample's MLA measure the group?
                     "abundance": by_group.get(group, 0.0)})

named_calls = pd.DataFrame(rows)

# calls the MLA can judge: how often is the named mineral group there at all?
judged = named_calls[named_calls["reported"]]
summary = judged.groupby("group")["abundance"].agg(
    n_calls="size",
    present=lambda a: (a > 0).mean(),                               # any amount
    present_1pct=lambda a: (a >= 0.01).mean(),                      # 1% or more
    median_abundance=lambda a: a[a > 0].median())                   # how much, when it's there
print(summary.sort_values("n_calls", ascending=False).round(3))
#            n_calls  present  present_1pct  median_abundance
#group                                                       
#chlorite        268     1.00         0.828             0.020
#white_mica      172     1.00         0.895             0.027
#smectite         47     0.83         0.170             0.003
#biotite           1     1.00         0.000             0.002

# calls the MLA can't judge: the named group isn't measured in this sample
print(named_calls.loc[~named_calls["reported"], "group"].value_counts())
#[paste output]
#%% H3: does Tetracorder's fit track mineral abundance?
rows = []
for x, y in np.ndindex(winners.shape):
    rule = winners[x, y]
    if not rule:
        continue                                        # no call: nothing to compare
    abundances = np.nan_to_num(mla[x, y])
    if abundances.sum() == 0 or abundances[exclude_mask].sum() > 0.7:
        continue                                        # no MLA data, or more than 70% Porosity, Unknown, Other

    # Tetracorder side: the mineral group(s) the winner names
    named = set(rule_lookup.loc[rule_lookup["rule_id"] == rule, "group"]) - {""}

    # MLA side: the abundance of those mineral group(s) in the pixel
    abundance = abundances[active_mask & np.isin(channel_group, list(named))].sum()

    rows.append({"group": ";".join(sorted(named)), "mixture": len(named) > 1, "abundance": abundance,
                 "fit": fit[x, y], "depth": depth[x, y], "fit_depth": fit_depth[x, y]})

calls = pd.DataFrame(rows)

# correct calls only: the winner names a single mineral group, and the MLA says that group is there
correct = calls[~calls["mixture"] & (calls["abundance"] > 0)]

# Spearman correlation with abundance: 1 = rises with abundance, 0 = unrelated
rho = correct.groupby("group")[["abundance", "fit", "depth", "fit_depth"]].corr(method="spearman")
rho = rho.xs("abundance", level=1)[["fit", "depth", "fit_depth"]]
rho.insert(0, "n_pixels", correct["group"].value_counts())
print(rho.round(2))
#            n_pixels   fit  depth  fit_depth
#group                                       
#biotite            1   NaN    NaN        NaN
#chlorite         263 -0.11   0.25       0.20
#smectite          39  0.58   0.62       0.62
#white_mica       172  0.28   0.41       0.40

# median fit, depth and fit × depth in each abundance bin
correct = correct.assign(bin=pd.cut(correct["abundance"], BINS, labels=LABELS))
for metric in ["fit", "depth", "fit_depth"]:
    print(f"\nmedian {metric} by abundance")
    print(correct.pivot_table(index="group", columns="bin", values=metric, aggfunc="median", observed=True).round(3))
# =============================================================================
# median fit by abundance
# bin          0-1%  1-2.5%  2.5-5%  5-10%  10-25%
# group                                           
# biotite     0.804     NaN     NaN    NaN     NaN
# chlorite    0.958   0.951   0.943  0.965     NaN
# smectite    0.708   0.857     NaN    NaN     NaN
# white_mica  0.893   0.921   0.933  0.930   0.962
# 
# median depth by abundance
# bin          0-1%  1-2.5%  2.5-5%  5-10%  10-25%
# group                                           
# biotite     0.090     NaN     NaN    NaN     NaN
# chlorite    0.091   0.086   0.098  0.106     NaN
# smectite    0.052   0.113     NaN    NaN     NaN
# white_mica  0.079   0.085   0.096  0.091   0.134
# 
# median fit_depth by abundance
# bin          0-1%  1-2.5%  2.5-5%  5-10%  10-25%
# group                                           
# biotite     0.072     NaN     NaN    NaN     NaN
# chlorite    0.085   0.082   0.093  0.102     NaN
# smectite    0.052   0.098     NaN    NaN     NaN
# white_mica  0.069   0.077   0.090  0.085   0.129
# =============================================================================

#%% Thiele et al. vocabulary: their 22 groups, their pixel mask, their thresholds
from scipy.ndimage import binary_erosion

CLIP = 2                                                          # edge erosion, in pixels
CLOSURE = 0.7                                                     # the 22 groups must add up to at least 70%
OCCURRENCE = 0.025                                                # a group is present above 2.5%

PAPER22 = sorted(set(mla_lookup["paper_group"]) - {""})
channel_paper = mla_lookup.loc[mineral_channels, "paper_group"].to_numpy()

# MLA side: the abundance of each of the 22 groups in every pixel, SWIR-active or not
filled = np.nan_to_num(mla)
paper_abundance = np.stack([filled[..., channel_paper == pg].sum(axis=-1) for pg in PAPER22], axis=-1)

# their mask: erode the sample edges, drop pixels where the 22 groups add up to under 70%, renormalise the rest
inside = binary_erosion((filled > 0).any(axis=-1), structure=np.ones((CLIP, CLIP)))
total = paper_abundance.sum(axis=-1)
keep = inside & (total >= CLOSURE)
paper_abundance = paper_abundance[keep] / total[keep, None]
print(f"{keep.sum()} of {keep.size} pixels kept")
#958 of 1395 pixels kept

# the two truths: every group over 2.5% (occurrence), and the single most abundant of all 22 (dominant phase)
occurrence = paper_abundance > OCCURRENCE
dominant = paper_abundance.argmax(axis=1)

# Tetracorder side: which of the 22 groups the winner names (a mixture names several; most rules name none)
rule_paper = {rule: set(groups) - {""} for rule, groups in rule_lookup.groupby("rule_id")["paper_group"]}
called = np.array([[pg in rule_paper.get(rule, set()) for pg in PAPER22] for rule in winners[keep]])

# yes/no per group, against each truth: precision, recall and MCC as in H1
for i, pg in enumerate(PAPER22):
    if not (occurrence[:, i].any() or called[:, i].any()):
        continue                                                  # never present and never called
    result = f"{pg:12s}"
    for name, truth in [("occurrence", occurrence[:, i]), ("dominant", dominant == i)]:
        c = called[:, i]
        tp, fp = (c & truth).sum(), (c & ~truth).sum()
        fn, tn = (~c & truth).sum(), (~c & ~truth).sum()
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denom if denom else np.nan
        result += f"  {name}: n={tp + fn:4d} precision={precision:.2f} recall={recall:.2f} MCC={mcc:.2f}"
    print(result)
#Biotite       occurrence: n= 388 precision=0.00 recall=0.00 MCC=-0.03  dominant: n= 101 precision=0.00 recall=0.00 MCC=-0.01
#Chlorite      occurrence: n= 258 precision=0.37 recall=0.38 MCC=0.14  dominant: n=   0 precision=0.00 recall=nan MCC=nan
#Epidote       occurrence: n=  68 precision=nan recall=0.00 MCC=nan  dominant: n=   0 precision=nan recall=nan MCC=nan
#Fluorite      occurrence: n=  20 precision=nan recall=0.00 MCC=nan  dominant: n=   0 precision=nan recall=nan MCC=nan
#Illite        occurrence: n=   0 precision=0.00 recall=nan MCC=nan  dominant: n=   0 precision=0.00 recall=nan MCC=nan
#K-feldspar    occurrence: n= 945 precision=nan recall=0.00 MCC=nan  dominant: n=  77 precision=nan recall=0.00 MCC=nan
#Muscovite     occurrence: n= 422 precision=0.71 recall=0.07 MCC=0.11  dominant: n=   0 precision=0.00 recall=nan MCC=nan
#Phosphate     occurrence: n=  24 precision=nan recall=0.00 MCC=nan  dominant: n=   0 precision=nan recall=nan MCC=nan
#Plagioclase   occurrence: n= 949 precision=nan recall=0.00 MCC=nan  dominant: n= 198 precision=nan recall=0.00 MCC=nan
#Quartz        occurrence: n= 958 precision=nan recall=0.00 MCC=nan  dominant: n= 582 precision=nan recall=0.00 MCC=nan
#Sulphate      occurrence: n=   0 precision=0.00 recall=nan MCC=nan  dominant: n=   0 precision=0.00 recall=nan MCC=nan

































