# flotation-benchmark

[![CI](https://github.com/sanjaygrr/flotation-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjaygrr/flotation-benchmark/actions/workflows/ci.yml)

A leakage-aware benchmark for soft sensors on the public iron-ore flotation
dataset "Quality Prediction in a Mining Process".

The dataset is widely used to predict `% Silica Concentrate` from plant data.
Two features of the data make it easy to get very high scores that do not
carry over to a real soft sensor:

1. `% Iron Concentrate` is a lab assay of the same concentrate sample as the
   target. It is strongly (negatively) correlated with silica, and it is not
   available before the silica result itself. The dataset author raised this
   point on the original dataset page.
2. The process variables are logged every 20 s but the lab assays are hourly,
   so each target value is repeated about 180 times. A random split of the
   20 s rows puts near-identical copies of the same hour in train and test.

This project measures how much each of these inflates the scores, and what is
left under evaluation protocols that respect time.

The plant is a reverse cationic flotation circuit: amine (`Amina Flow`)
collects quartz into the froth, starch depresses the iron oxides, and the
iron concentrate leaves with the column underflow. `% Iron Concentrate` and
`% Silica Concentrate` are two assays of that final concentrate. They are not
a closed pair (other gangue minerals remain), so their correlation is −0.80
rather than −1.

## Results

Full run on the OpenML file (737,453 rows), default settings, seed 0.
R² on the test set, mean ± standard deviation across folds (B and B1 are a
single split):

| Protocol | Mean | Persistence | Ridge | Random forest | LightGBM |
|---|---|---|---|---|---|
| A. Naive (leaky) | -0.00 ± 0.00 | 0.48 ± 0.00 | 0.68 ± 0.00 | **0.98 ± 0.00** | 0.90 ± 0.00 |
| A1. Naive without iron | -0.00 ± 0.00 | 0.48 ± 0.00 | 0.15 ± 0.00 | 0.81 ± 0.00 | 0.65 ± 0.00 |
| A2. Hourly, shuffled | -0.00 ± 0.00 | 0.48 ± 0.02 | 0.16 ± 0.01 | 0.34 ± 0.02 | 0.36 ± 0.04 |
| B. Time-ordered split | -0.00 | **0.54** | -0.10 | 0.09 | 0.05 |
| B1. Time-ordered, with iron | -0.00 | 0.54 | 0.61 | 0.61 | 0.60 |
| C. Walk-forward CV | -0.24 ± 0.34 | **0.45 ± 0.09** | -0.17 ± 0.12 | -0.30 ± 0.56 | -0.35 ± 0.46 |
| C1. Walk-forward, no feed assays | -0.24 ± 0.34 | **0.45 ± 0.09** | -0.15 ± 0.14 | -0.24 ± 0.53 | -0.30 ± 0.56 |
| D. One-hour-ahead forecast | -0.25 ± 0.36 | 0.45 ± 0.09 | 0.47 ± 0.10 | 0.40 ± 0.14 | 0.37 ± 0.15 |

MAE in % SiO₂ (mean across folds):

| Protocol | Mean | Persistence | Ridge | Random forest | LightGBM |
|---|---|---|---|---|---|
| A. Naive (leaky) | 0.916 | 0.510 | 0.492 | **0.108** | 0.257 |
| A1. Naive without iron | 0.916 | 0.510 | 0.820 | 0.332 | 0.497 |
| A2. Hourly, shuffled | 0.917 | 0.510 | 0.815 | 0.707 | 0.681 |
| B. Time-ordered split | 0.946 | **0.503** | 0.951 | 0.901 | 0.903 |
| B1. Time-ordered, with iron | 0.946 | 0.503 | 0.562 | 0.582 | 0.578 |
| C. Walk-forward CV | 0.983 | **0.494** | 0.925 | 1.001 | 0.999 |
| C1. Walk-forward, no feed assays | 0.983 | **0.494** | 0.905 | 0.967 | 0.966 |
| D. One-hour-ahead forecast | 0.983 | **0.494** | 0.573 | 0.652 | 0.637 |

MAE is the headline metric: it is less sensitive to the few large assay
jumps than RMSE and R². In walk-forward CV each fold is refitted on all past
hours and then scores a block of about 680 hours without refitting inside the
block. Fold-level R² varies a lot (random forest: −0.15, −1.29, −0.21, 0.00,
0.12; the worst fold is 2017-05-19 to 2017-06-16), so the means above hide a
wide spread; `results/scores.csv` and the report list every fold.

What this shows, on this dataset and with these models:

- **The naive setup is badly inflated.** Random forest reaches R² 0.98 under
  A, close to the R² 0.983 that Loseto et al. (2022) report with randomly
  sampled 20 s rows (see Related work). Under walk-forward CV (C) the same
  model has R² −0.30 and its MAE grows from 0.108 to 1.001.
- **The iron column is only part of the inflation.** Dropping iron (A → A1)
  lowers random forest R² from 0.98 to 0.81. Collapsing to hourly means
  (A1 → A2) lowers it to 0.34, and respecting time order (A2 → B, C) takes it
  to about zero or below. The A1 → A2 step mixes two things: it removes the
  duplicated labels, but it also averages away the 20 s detail and shrinks
  the sample. A time-ordered protocol with richer within-hour features is
  planned to separate them.
- **With the iron column, a time-ordered split gives R² about 0.6** (B1 vs
  B), roughly the 0.64 one would expect from the −0.80 correlation alone. It
  is a co-assay of the same sample as the target, so this measures the leak,
  not a useful sensor. Even so, B1 models lose to persistence in MAE
  (0.562 vs 0.503).
- **Process data alone do not beat the last lab value.** Under B, C and C1,
  persistence (the previous hour's assay) has MAE about 0.49 to 0.50, while
  every learned model is above 0.9, close to the training mean. Walk-forward
  R² of the learned models is negative in most folds.
- **With the recent assays as inputs (D), models roughly tie persistence.**
  Ridge has slightly lower RMSE (0.769 vs 0.781) and higher R² (0.47 vs
  0.45), within the fold spread, but higher MAE in every fold (mean 0.573 vs
  0.494). Permutation importance puts almost all of its skill on the two
  lagged assays.
- Removing the feed assays (C1) changes little; they are not what holds the
  honest models back.
- **Repeated and interpolated hours flatter persistence, but do not explain
  the result.** The 446 hours that repeat the previous value give persistence
  zero error. Rescoring (without refitting) only the 2,799 walk-forward test
  hours with a fresh, non-interpolated assay gives MAE 0.551 for persistence
  against 0.862 for ridge and 0.958 for LightGBM (0.586 for ridge under D).

So the hypothesis holds here: scores reported under random splits of the
20 s rows, especially with the iron column, say little about how a soft
sensor would perform on future data from this plant. The honest question for
this dataset is whether any model can beat persistence, and in v0.1 none does
by a clear margin.

The HTML report (`flotation-benchmark report`) has the same tables plus
fold-level spread, the naive-to-honest progression, feature importances and
a plot of walk-forward predictions.

## Protocols

All protocols use the same models and the same metrics (MAE, RMSE and R²,
computed on the test set; mean and standard deviation across folds).

| Key | Name | Rows | Split | Iron column |
|---|---|---|---|---|
| `naive` | A. Naive | 20 s | shuffled 5-fold | yes |
| `rows_random_no_iron` | A1. Naive without iron | 20 s | shuffled 5-fold | no |
| `hourly_random_no_iron` | A2. Hourly, shuffled | hourly | shuffled 5-fold | no |
| `time_holdout` | B. Time-ordered split | hourly | first 80 % / last 20 % of hours | no |
| `time_holdout_with_iron` | B1. Time-ordered, with iron | hourly | as B | yes |
| `walk_forward` | C. Walk-forward CV | hourly | expanding window, 5 test blocks, 24 h gap | no |
| `walk_forward_no_feed` | C1. Walk-forward, no feed assays | hourly | as C, also without the feed assays | no |
| `forecast` | D. One-hour-ahead forecast | hourly | as C | no |

A reproduces the common setup. A1, A2 and B1 are ablations that remove or add
one leak at a time. B, C, C1 and D are the leakage-aware protocols.

- **Hourly rows.** Process variables are averaged over each hour. The target
  is averaged too; it is constant within most hours (see data notes).
- **Walk-forward CV.** The sorted hours are cut into 6 consecutive blocks.
  Fold *i* tests on block *i*+1 and trains on every hour that ends at least
  24 h before the test block starts. Training only ever sees the past.
- **Forecast.** The input for target hour *t*+1 is the process means of hours
  *t* and *t*−1 plus the silica assays of hours *t* and *t*−1. Rows whose
  inputs or target cross a gap in the data are dropped. This assumes the
  assay of hour *t* is available by the end of hour *t*; the dataset does not
  record the real lab turnaround.

### Models

- **Mean**: training-set mean.
- **Persistence**: the last available lab value (previous hour; for D, the
  current hour). This is what an operator already knows without any model.
- **Ridge** (standardized inputs, alpha chosen by internal CV).
- **Random forest** (100 trees, min 3 samples per leaf, 33 % of features per
  split, at most 50 000 bootstrap rows per tree).
- **LightGBM** (300 trees, learning rate 0.05, 31 leaves, row and column
  subsampling 0.8).

Settings are kept light so a full run fits on an 8 GB laptop. No tuning was
done per protocol. A small recurrent network was left out of v0.1.

## Data

- Source: OpenML dataset [43311](https://www.openml.org/d/43311), a mirror of
  the [Kaggle dataset](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)
  by Eduardo Magalhães Oliveira. OpenML lists the license as CC0 (public
  domain). The mirror is used because it can be downloaded without an account.
- File: `https://data.openml.org/datasets/0004/43311/dataset_43311.pq`
  (65 MB parquet), sha256
  `2f0675981a04e2747cfce0d5c16ea56d3f1d14817fcfcd7a96c469a020246f31`.
  `flotation-benchmark fetch` checks it and writes `data/raw/SOURCE.json`.
  The data are not stored in this repository.
- Every column is stored as text with a decimal comma (`"55,2"`). The `date`
  column has hourly resolution only; the rows that share a stamp are the 20 s
  samples of that hour, in file order.

Facts checked on the downloaded file (also printed in the report):

- 737,453 rows; 4,097 distinct hours from 2017-03-10 01:00 to 2017-09-09
  23:00. 4,095 hours have 180 rows; 2017-03-10 01:00 has 174 and
  2017-04-10 00:00 has 179.
- One gap: no rows between 2017-03-16 05:00 and 2017-03-29 12:00
  (318 missing hours, about 13 days).
- In 310 hours `% Silica Concentrate` changes inside the hour as a linear
  ramp between two assay values, i.e. it was interpolated in the published
  file. In 446 hours it repeats the previous hour's value exactly. Hourly
  aggregation takes the mean (the midpoint of a ramp). Lagged inputs and the
  persistence baseline use a causal series in which interpolated hours are
  replaced by the last real value, since an interpolated value depends on the
  next assay.
- `% Iron Feed` and `% Silica Feed` change only 308 times over the period
  (roughly every 13 hours), so they are not hourly measurements.
- Correlation between `% Iron Concentrate` and `% Silica Concentrate` is
  −0.80 (both hourly and on 20 s rows). The lag-1 autocorrelation of hourly
  silica is 0.77, which is why persistence is a strong baseline.

## Usage

Requires Python 3.10 or newer.

```bash
git clone https://github.com/sanjaygrr/flotation-benchmark
cd flotation-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

flotation-benchmark fetch     # download and verify the data (65 MB)
flotation-benchmark run       # all protocols and models, writes results/
flotation-benchmark report    # writes results/report.html
```

Useful options for `run`: `--quick` (every 10th 20 s row, no permutation
importances), `--protocols walk_forward forecast`, `--models persistence
lightgbm`, `--gap-hours 48`, `--folds 5`, `--jobs 4`. The full run takes
about 6 minutes (peak memory about 750 MB) on an Apple M3 laptop with 8 GB RAM; most of the
time goes to the two 20 s protocols.

Outputs in `results/`:

| File | Content |
|---|---|
| `scores.csv` | one row per protocol, model and fold, with train and test date ranges |
| `summary.csv` | mean and standard deviation across folds |
| `predictions.csv` | test-set predictions for the hourly protocols |
| `importances.csv` | permutation importances under B, C and D |
| `meta.json` | configuration, package versions, data facts and checksum |
| `report.html` | self-contained report |

Tests run without network access, on a small fixture and synthetic data:

```bash
pytest -q
```

They cover parsing, hourly aggregation, gap detection, the forecast table
(inputs strictly before the target hour), split leakage checks (no test
stamp inside or within the gap of the training range) and the metrics.

## Related work

Papers that use this dataset. Columns only record what the paper states.
"not stated" means the text we read does not say; "not checked" means we
could not access the full text.

| Paper | Split | `% Iron Concentrate` as input | Time resolution | Reported silica score | What we read |
|---|---|---|---|---|---|
| Doğan, Birant, Kut (2019). Multi-target regression for quality prediction in a mining process. UBMK 2019. [doi:10.1109/UBMK.2019.8907120](https://doi.org/10.1109/UBMK.2019.8907120) | not stated in abstract | no: iron and silica are predicted jointly as targets | not stated in abstract | R² used; no silica value in the abstract | abstract only |
| Loseto et al. (2022). Osmotic cloud-edge intelligence for IoT-based cyber-physical systems. *Sensors* 22(6), 2166. [doi:10.3390/s22062166](https://doi.org/10.3390/s22062166) | samples extracted randomly; 1/7 held out for validation | not stated | 20 s rows (737,454 samples) | R² 0.983, MSE 0.021 | full text |
| Ahmed, Pourrahimian (2024). Quality prediction in process of flotation plant with gradient boosting and LSTM model. MOL Report, University of Alberta. [PDF](https://sites.ualberta.ca/MOL/DataFiles/2024_Papers/501_Ahmed.pdf) | 70 % train, 30 % split into test and validation; ordering not stated | no: iron is a second output for XGBoost, and dropped for the LSTM | 2-minute intervals (medians of 6 rows for the LSTM) | XGBoost test R² 0.91; LSTM test R² 0.57 | full text |
| Dong, Gong, He (2026). A new multilayer tree structure belief rule base-based prediction method for key indicators of flotation process. *PLOS ONE* 21(2). [doi:10.1371/journal.pone.0336336](https://doi.org/10.1371/journal.pone.0336336) | chronological (first 70 % train, last 30 % test); the text also describes training on March and testing on April hourly data | yes (listed as input X9) | hourly | not checked | full text; the target is described as both tailings silica and concentrate silica |
| Ortega-Moody, Ward, Jenab, Isaza (2026). An LSTM approach for quality prediction in a mining process using ensemble data interpolation. *Applied Sciences* 16(7), 3419. [doi:10.3390/app16073419](https://doi.org/10.3390/app16073419) | 70/10/20 split of 3,948 hourly samples (§3.3); random or chronological not stated | no, removed (§3.2) | hourly for the model; minute-level interpolated estimates | LSTM MAE 0.499, RMSE 0.705, R² 0.631 (Table 6) | full text (open access, CC BY) |
| Osei (2019). Machine learning-based quality prediction in the froth flotation process of mining. MSc thesis, Dalarna University. [PDF](https://www.diva-portal.org/smash/get/diva2%3A1386720/FULLTEXT01.pdf) | random: 70% train with 10-fold CV, 30% test; "randomly sampled" (§3.1.6) | no: "predict silica concentrate without the use of iron concentrate" (§3.1.1) | 20 s rows | best neural network MAE 0.381, RMSE 0.554 (Table 11) | full text |
| Basilio, Diniz, Guimarães, Silva (2026). Silica grade forecast in flotation processes: evaluation and statistical diagnosis of machine learning methods. SBSI 2026 (extended). [link](https://sol.sbc.org.br/index.php/sbsi_estendido/article/view/42009) | one-step-ahead (T+1) forecast | not stated in abstract | not stated in abstract | public dataset: random forest R² 0.66, LSTM R² 0.67 | abstract only |
| Xie, Qiu (2022). Robust monitoring of multivariate processes with short-ranged serial data correlation. *Quality and Reliability Engineering International*. [doi:10.1002/qre.3199](https://doi.org/10.1002/qre.3199) | chronological: first 300 observations as in-control, next 100 monitored (§4) | no (pulp flow, air flow, levels only) | not stated | n/a, process monitoring, not silica prediction | full text |
| Hofer, Krempl, Lang (2024). Anticipative Bayesian classification for data streams with verification latency. *Journal of Applied Statistics*. [doi:10.1080/02664763.2024.2319222](https://doi.org/10.1080/02664763.2024.2319222) | sequential test-then-train stream | not stated | not stated | n/a, target turned into a classification stream (AUC, F1) | full text |
| Caers, Eskanlou (2026). Causal analysis of an industrial iron-ore flotation circuit: which setpoint effects can be identified from plant data. *Minerals Engineering*. [doi:10.1016/j.mineng.2026.110922](https://doi.org/10.1016/j.mineng.2026.110922) | n/a, causal analysis on 4,097 hourly records | not checked | hourly | n/a, no predictive score | abstract only |

The scores above come from different splits, preprocessing and test sets, so
they are not directly comparable with each other or with this benchmark. The
row-level random-split result of Loseto et al. (R² 0.983) is close to protocol
A here (random forest R² 0.976). Corrections and additions are welcome.


None of the papers above that report a silica score also report a
persistence baseline. For reference, persistence gets MAE 0.494 to 0.503
and R² 0.45 to 0.54 under the time-ordered protocols here (B, C, C1, D).
Osei (2019) drops the iron column but splits the 20 s rows at random.
Ortega-Moody et al. (2026) use hourly data without the iron column, and
their LSTM (MAE 0.499, R² 0.631) is in the same range as persistence here;
they do not state whether the split was chronological. These scores are not
directly comparable with this benchmark, and no claim is made that they are
wrong.

## Background

**Process.** The data come from a reverse cationic flotation circuit for
iron ore in Brazil. Quartz is floated with an amine collector, the iron
oxides are depressed with starch, and the pulp is alkaline (pH about 9 to
10.5). The concentrate, which is the product, is the column underflow,
the sunk stream. `% Silica Concentrate` is the SiO₂ left in
that product, the impurity the circuit exists to remove
([Filippov et al., 2014](https://doi.org/10.1016/j.minpro.2014.01.002);
[Lima et al., 2013](https://doi.org/10.1016/j.mineng.2013.03.001);
[Araujo et al., 2005](https://doi.org/10.1016/j.mineng.2004.08.023)).

**Why a soft sensor.** Grade comes from laboratory assays that are
infrequent and delayed, while airflow, levels, reagent dosages and pH are
logged every few seconds. Data-driven soft sensors try to bridge that gap
([Kadlec et al., 2009](https://doi.org/10.1016/j.compchemeng.2008.12.012);
[Jovanović et al., 2015](https://doi.org/10.1016/j.mineng.2015.09.020);
[McCoy and Auret, 2019](https://doi.org/10.1016/j.mineng.2018.12.004);
[Aldrich et al., 2010](https://doi.org/10.1016/j.minpro.2010.04.005)).
The dataset records when each assay was sampled, not when its result
reached the control room. Reported sampling-plus-assay delays for flotation
labs range from tens of minutes to hours (e.g. 30 to 180 min for a zinc
circuit, [Zhang et al., 2026](https://doi.org/10.1016/j.mineng.2026.110816)),
so the forecasting protocol (D) is a best case.

**Why persistence is hard to beat.** Hourly concentrate grade moves slowly,
so the last assay already carries most of the information about the next
one. Any soft sensor for this kind of target should be compared against it.

**Other open data.** No open industrial flotation dataset of this kind
(continuous process signals plus concentrate assays) for copper was found.
Open flotation data that do exist are mostly froth images, for example
[Yantén et al., 2023](https://doi.org/10.3390/data8040069).

## Limitations

- One plant, one period (about five months of usable data). Results may not
  transfer to other circuits or ores.
- Light, untuned models. A tuned model, longer lag windows or a recurrent
  network could do better, especially under D. The point of v0.1 is the
  protocol, not the best possible model.
- The real lab turnaround and sampling times are not in the data. Protocol D
  assumes the assay of hour *t* is known by the end of hour *t*; if the lab is
  slower, D is optimistic. The same applies to the feed assays.
- The target itself was partly interpolated and forward-filled before
  publication; the benchmark cannot recover the original assay times.
- Air flows and levels are controlled variables (the level set point is the
  froth/pulp interface). A same-hour fit learns the closed-loop operating
  policy, which can drift; that is one plausible reason the process-only
  models fail under C.
- The 24 h gap in walk-forward CV is a choice, not derived from the process
  dynamics. `--gap-hours` changes it.
- Fold scores are unweighted means over folds of near-equal size; B and B1
  are single splits with no spread.
- The related-work table only covers papers whose text we could reach; it is
  not a systematic review.

## Citation

If you use this benchmark, please cite the repository and the original
dataset. A `CITATION.cff` will be added with the first tagged release.

## License

Code: MIT (see `LICENSE`). Data: not redistributed here; see the source above.
