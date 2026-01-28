# Soy Fermentation Metabolomics Analysis

Analysis pipeline for identifying day-specific metabolite enrichment patterns during soy fermentation using PLS-DA and statistical testing.

## Overview

This project analyzes metabolomics data from soy fermentation samples collected at different time points (Day 1, 10, 28, and 60) to identify metabolites that change significantly over the fermentation process.

## Features

- **Data Integration**: Links metadata (sample info, time points, replicates) with metabolite abundance matrix
- **PLS-DA Analysis**: Partial Least Squares Discriminant Analysis to identify day-specific enrichment patterns
- **VIP Scores**: Variable Importance in Projection scores to rank metabolite discriminatory power
- **Statistical Testing**:
  - One-way ANOVA with FDR correction (Benjamini-Hochberg) for multiple testing
  - Post-hoc Tukey HSD for pairwise day comparisons
- **Visualizations**: PLS-DA scores plots, time course line plots, and individual metabolite profiles with significance annotations

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
scipy
statsmodels
```

Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn scipy statsmodels
```

## Usage

```bash
python analyze_metabolites.py
```

## Input Files

Place input files in the `input/` directory:

| File | Description |
|------|-------------|
| `soy-fermentation_input.txt` | Tab-separated matrix with samples (rows) × metabolites (columns) |
| `Soy-fermentation_metadata.txt` | Tab-separated metadata with Sample, Days, and Rep columns |

## Output Files

Generated in the `output/` directory:

| File | Description |
|------|-------------|
| `vip_scores.csv` | VIP scores for all metabolites |
| `anova_results.csv` | ANOVA F-statistics, p-values, and FDR-adjusted p-values |
| `tukey_posthoc_results.csv` | Pairwise Tukey HSD comparison results |
| `plsda_scores_plot.png` | PLS-DA scores visualization (PC1 vs PC2, PC1 vs PC3) |
| `metabolites_time_course.png` | Line plot of top 20 discriminating metabolites over time |
| `metabolites_individual_plots.png` | Individual metabolite plots with significance annotations |
| `metabolite_summary_stats.csv` | Summary statistics (mean, std) per metabolite per time point |

## Results Summary

- **573 metabolites** analyzed across 24 samples (4 time points × 6 replicates)
- **263 metabolites** significant by ANOVA (p_adj < 0.05)
- **175 metabolites** highly significant (p_adj < 0.001)
- **100% cross-validation accuracy** in PLS-DA classification

### Top Discriminating Metabolites

| Metabolite | ANOVA F-statistic | p_adjusted | VIP Score |
|------------|-------------------|------------|-----------|
| Guanine | 2166.2 | 1.57e-22 | 1.51 |
| N-[(2S)-2-Amino-2-carboxyethyl]-L-glutamate | 1269.5 | 1.60e-20 | 1.86 |
| Ketoleucine | 1145.2 | 2.97e-20 | 1.66 |
| Baicalein | 930.8 | 1.75e-19 | 1.79 |
| Naringenin | 847.6 | 3.54e-19 | 1.72 |

## Significance Notation

- `***` p < 0.001
- `**` p < 0.01
- `*` p < 0.05
- `ns` not significant

## License

MIT
