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

A two-group test chosen for more than two groups runs on each pair,
Holm-corrected. With a **Group / Color By** column, colours are compared
within each X group, and p-values are also Holm-corrected across X groups.
Reported for each comparison: adjusted and unadjusted p, the difference
with its 95% CI (where the test gives one) and an effect size (Cohen's d
and Hedges' g, rank-biserial r, Cohen's dz; eta² / epsilon² for omnibus
tests). Stars: ns p ≥ 0.05, * < 0.05, ** < 0.01, *** < 0.001,
**** < 0.0001; all tests two-sided.

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
