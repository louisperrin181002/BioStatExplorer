# BioStat Explorer

Desktop app for plate-reader and other lab data: import, filter, plot with
significance stars, statistics, and publication exports.

## Run

```
pip install -r requirements.txt
python run.py
```

Build the Mac app: `pyinstaller --noconfirm BioStatExplorer.spec`
(Windows: the GitHub Actions workflow builds an .exe on every push to main).

## Getting data in

* **Open Data** (⌘O) — Excel (.xlsx), CSV/TSV. One row per sample, one
  column per measurement or label. Merged cells (e.g. *Experiment* written
  once over its replicate rows) are filled into every row; instrument
  preambles above the table are skipped; numbers stored as text are
  converted. Several files are stacked, with a *Source File* column.
* **Import Plate** (⇧⌘I) — plate-reader exports (e.g. BMG CLARIOstar).
  Every 8 × 12 (or other) grid in the sheet is found — GFP, mKate, OD, raw
  and normalised — and named from its wavelengths. Wells at the detector
  maximum (saturation) and overflow cells are flagged. Add a plate map
  (strain / medium per well: type, paste from Excel or import it), then
  **Build Dataset From Plate**.

## Projects and undo

**File → Save Project** (⌘S) keeps everything in one `.bse` file: the data,
every graph setting (colours, order, labels, filters, test), the plate
grids and plate series, the curve settings and the figure layout. **Open
Project** (⇧⌘O) brings it back exactly. **Edit → Undo / Redo** (⌘Z / ⇧⌘Z)
for data changes, colours, order, labels and layout. Closing with unsaved
changes asks to save.

## Plates: corrections and series

* **Corrections** (Plate tab): subtract blanks — the wells labelled
  `blank` (or any label, several separated by commas) in a plate-map
  layer, optionally matched by medium so each well gets its own medium's
  blank — and divide every readout by OD. Raw values are kept as
  `… (raw)`; blank wells are dropped.
* **Plate series**: select several export files in Import Plate (e.g. 8 h,
  16 h, 24 h). The time is read from the file names (`16h`, `30 min`,
  `3 days`; editable), one plate map applies to all, and the dataset gets
  `Plate` and `Time (h)` columns. Import the OD files of the same time
  points afterwards: they are added to the same plates. Corrections are
  applied plate by plate.

## Curves

Curve fitting lives on the **Plotting Dashboard** — chart type *Growth
curve (mean ± error)* — rather than in a tab of its own, so a fitted
curve is built with the same controls as any other figure. It fits, per
well or per group: a logistic growth curve (K, growth rate r, doubling
time, lag time, time of fastest growth), the model-free maximum growth
rate (steepest slope of ln OD), or a dose-response curve (4PL: EC50 with
95% CI, Hill slope, bottom, top). **Compare Parameters on the Plotting
Dashboard** turns the table into the dataset, e.g. growth rate by strain
with statistics (Undo returns to the data).

How the error is drawn is set in the Style tab: **Error bars**
(SD / SEM / 95% CI), **Shown as** (band, bars or mean only), **Error
around** (the fitted curve or the mean — the band is clamped to enclose
whichever is chosen) and **Draw the error as smooth curves**, which
draws the band on the fitted curve's own grid so its edges follow the
curve instead of cutting between timepoints. Only the drawing changes:
the error is always the replicate spread, interpolated between
timepoints with a monotone cubic that cannot overshoot.

## Figure layout

**＋ Add to Layout** (above the graph) puts the current graph on the
**Layout** page: journal width (85 / 120 / 174 / 190 mm), columns, panel
height, one font size, panel letters. Each panel keeps its own data, so
graphs from different datasets share a page. Export PDF / SVG / PNG / TIFF.

## FlowPy statistics

Open a FlowPy *Export Data* table (CSV or Excel) like any other file: one
row per sample with columns such as `Lymph / CD3+ | CD3 FITC-A Median`; X
defaults to your Group column. Long tables (one row per sample ×
population) show one population at a time (*Show only › Population*).

## Statistics

Both the **Statistical Analysis** report and the stars on plots come from
the same engine (`src/statistics_engine.py`), so they always agree.

| Data | Test | Comparisons |
|---|---|---|
| 2 groups | Student's t, Welch's t, Mann-Whitney U | — |
| > 2 groups, normal | one-way ANOVA | Tukey HSD (all pairs) or Dunnett (vs a reference group) |
| > 2 groups, unequal variances | Welch ANOVA | Games-Howell |
| > 2 groups, not normal | Kruskal-Wallis | Dunn's, Bonferroni over the comparisons made |
| paired, 2 groups | paired t, Wilcoxon signed-rank | — |
| paired, > 2 groups | repeated-measures ANOVA, Friedman | paired tests, Holm-corrected |
| two factors (X × Color by) | two-way ANOVA (type II), partial eta² | within each X group, pooled error, Šídák |

A two-group test chosen for more than two groups runs on each pair,
Holm-corrected. With a **Group / Color By** column, colours are compared
within each X group, and p-values are also Holm-corrected across X groups.
Reported for each comparison: adjusted and unadjusted p, the difference
with its 95% CI (where the test gives one) and an effect size (Cohen's d
and Hedges' g, rank-biserial r, Cohen's dz; eta² / epsilon² for omnibus
tests). Stars: ns p ≥ 0.05, * < 0.05, ** < 0.01, *** < 0.001,
**** < 0.0001; all tests two-sided.

**Outliers**: Grubbs' test (alpha 0.05) within each group; flagged values
are listed in the report, and *Leave out outliers* excludes them from the
test and the plot.

**Auto** checks normality (Shapiro-Wilk, per group, n ≥ 3) and equal
variances (Brown-Forsythe, n ≥ 3) and picks the test from the table. With
fewer values it assumes normality and says so. These checks have little
power with few replicates, so if you know your data's distribution, choose
the test yourself.

Example methods text (adapt to your choices):

> Groups were compared by one-way ANOVA followed by Dunnett's multiple
> comparisons test against the wild type (BioStat Explorer, using SciPy
> and statsmodels). Data are mean ± SEM of n biological replicates;
> p < 0.05 was considered significant.

The engine is checked against SciPy, statsmodels, pingouin and
scikit-posthocs: `pip install -r requirements-dev.txt` then
`python tests/test_statistics.py`.

## Exports

* Figure: PNG / TIFF (600 dpi), SVG and PDF with editable text (Inkscape,
  Illustrator).
* Statistics (Excel): group summary, every comparison as a table, and the
  full report. PDF report (A4) with the figure, summary and report.
* Data table (CSV) of the rows currently shown.
