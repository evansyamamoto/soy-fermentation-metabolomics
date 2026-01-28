#!/usr/bin/env python3
"""
Metabolite Analysis Script for Soy Fermentation Data
- Links metadata to input matrix using Sample names
- Identifies Day-specific enrichment patterns using PLS-DA
- Performs ANOVA with post-hoc Tukey HSD for statistical significance
- Visualizes individual metabolites grouped by time in line plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import warnings
import os

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Create output directory
os.makedirs('output', exist_ok=True)

# ============================================================================
# 1. Load and Link Data
# ============================================================================
print("=" * 60)
print("1. Loading and linking data...")
print("=" * 60)

# Load input matrix
input_data = pd.read_csv('input/soy-fermentation_input.txt', sep='\t')
print(f"Input matrix shape: {input_data.shape}")

# Load metadata
metadata = pd.read_csv('input/Soy-fermentation_metadata.txt', sep='\t')
print(f"Metadata shape: {metadata.shape}")

# Merge data using Sample names
merged_data = pd.merge(metadata, input_data, on='Sample', how='inner')
print(f"Merged data shape: {merged_data.shape}")
print(f"Days in dataset: {sorted(merged_data['Days'].unique())}")
print(f"Samples per day: {merged_data.groupby('Days').size().to_dict()}")

# Separate features and metadata
metabolite_columns = [col for col in merged_data.columns if col not in ['Sample', 'Days', 'Rep']]
X = merged_data[metabolite_columns].values
y = merged_data['Days'].values

print(f"\nNumber of metabolites: {len(metabolite_columns)}")

# ============================================================================
# 2. PLS-DA Analysis for Day-Specific Enrichment Patterns
# ============================================================================
print("\n" + "=" * 60)
print("2. PLS-DA Analysis for Day-Specific Enrichment Patterns")
print("=" * 60)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode days as categorical for PLS-DA
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Fit PLS-DA model (using 3 components)
n_components = 3
pls = PLSRegression(n_components=n_components)
pls.fit(X_scaled, y_encoded)

# Get scores for visualization
X_scores = pls.transform(X_scaled)

# Cross-validation prediction
y_pred_cv = cross_val_predict(pls, X_scaled, y_encoded, cv=5)
y_pred_cv_rounded = np.round(y_pred_cv).astype(int)
y_pred_cv_rounded = np.clip(y_pred_cv_rounded, 0, len(le.classes_) - 1)
cv_accuracy = accuracy_score(y_encoded, y_pred_cv_rounded)
print(f"Cross-validation accuracy: {cv_accuracy:.2%}")

# Calculate VIP scores for feature importance
def calculate_vip(pls_model, X, y):
    """Calculate Variable Importance in Projection (VIP) scores."""
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_

    p, h = w.shape
    vip = np.zeros(p)

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * (s.T @ weight).item() / total_s)

    return vip.flatten()

vip_scores = calculate_vip(pls, X_scaled, y_encoded)
vip_df = pd.DataFrame({
    'Metabolite': metabolite_columns,
    'VIP_Score': vip_scores
}).sort_values('VIP_Score', ascending=False)

# Save VIP scores
vip_df.to_csv('output/vip_scores.csv', index=False)
print(f"\nTop 10 metabolites by VIP score:")
print(vip_df.head(10).to_string(index=False))

# Identify significant metabolites (VIP > 1 is typically considered important)
significant_metabolites = vip_df[vip_df['VIP_Score'] > 1]['Metabolite'].tolist()
print(f"\nNumber of significant metabolites (VIP > 1): {len(significant_metabolites)}")

# ============================================================================
# Plot PLS-DA Scores
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color map for days
days_unique = sorted(merged_data['Days'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(days_unique)))
day_colors = {day: colors[i] for i, day in enumerate(days_unique)}

# Plot 1: PC1 vs PC2
ax1 = axes[0]
for day in days_unique:
    mask = merged_data['Days'] == day
    ax1.scatter(X_scores[mask, 0], X_scores[mask, 1],
                c=[day_colors[day]], label=f'Day {day}', s=100, alpha=0.8, edgecolors='black')
ax1.set_xlabel('PLS Component 1', fontsize=12)
ax1.set_ylabel('PLS Component 2', fontsize=12)
ax1.set_title('PLS-DA Scores Plot (PC1 vs PC2)', fontsize=14)
ax1.legend(title='Days', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: PC1 vs PC3
ax2 = axes[1]
for day in days_unique:
    mask = merged_data['Days'] == day
    ax2.scatter(X_scores[mask, 0], X_scores[mask, 2],
                c=[day_colors[day]], label=f'Day {day}', s=100, alpha=0.8, edgecolors='black')
ax2.set_xlabel('PLS Component 1', fontsize=12)
ax2.set_ylabel('PLS Component 3', fontsize=12)
ax2.set_title('PLS-DA Scores Plot (PC1 vs PC3)', fontsize=14)
ax2.legend(title='Days', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/plsda_scores_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nPLS-DA scores plot saved to: output/plsda_scores_plot.png")

# ============================================================================
# 3. ANOVA Statistical Testing
# ============================================================================
print("\n" + "=" * 60)
print("3. ANOVA Statistical Testing")
print("=" * 60)

# Perform one-way ANOVA for each metabolite
anova_results = []
for metabolite in metabolite_columns:
    # Get values for each day group
    groups = [merged_data.loc[merged_data['Days'] == day, metabolite].values
              for day in days_unique]

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    anova_results.append({
        'Metabolite': metabolite,
        'F_statistic': f_stat,
        'p_value': p_value
    })

anova_df = pd.DataFrame(anova_results)

# Apply FDR correction (Benjamini-Hochberg)
_, p_adjusted, _, _ = multipletests(anova_df['p_value'], method='fdr_bh')
anova_df['p_adjusted'] = p_adjusted
anova_df['Significant'] = anova_df['p_adjusted'] < 0.05

# Sort by p-value
anova_df = anova_df.sort_values('p_value')

# Merge with VIP scores
anova_vip_df = pd.merge(anova_df, vip_df, on='Metabolite')
anova_vip_df = anova_vip_df.sort_values('p_adjusted')

# Save ANOVA results
anova_vip_df.to_csv('output/anova_results.csv', index=False)

print(f"\nANOVA Results Summary:")
print(f"  - Total metabolites tested: {len(metabolite_columns)}")
print(f"  - Significant (p_adj < 0.05): {anova_df['Significant'].sum()}")
print(f"  - Significant (p_adj < 0.01): {(anova_df['p_adjusted'] < 0.01).sum()}")
print(f"  - Significant (p_adj < 0.001): {(anova_df['p_adjusted'] < 0.001).sum()}")

print(f"\nTop 10 metabolites by ANOVA p-value:")
print(anova_vip_df[['Metabolite', 'F_statistic', 'p_value', 'p_adjusted', 'VIP_Score']].head(10).to_string(index=False))

# ============================================================================
# Post-hoc Tukey HSD Test for Top Metabolites
# ============================================================================
print("\n" + "-" * 40)
print("Post-hoc Tukey HSD Analysis (Top 20 metabolites)")
print("-" * 40)

# Select top metabolites for post-hoc analysis
top_metabolites_anova = anova_vip_df.head(20)['Metabolite'].tolist()

posthoc_results = []
for metabolite in top_metabolites_anova:
    # Prepare data for Tukey HSD
    values = merged_data[metabolite].values
    groups = merged_data['Days'].astype(str).values

    # Perform Tukey HSD
    tukey = pairwise_tukeyhsd(values, groups, alpha=0.05)

    # Extract results
    for i in range(len(tukey.summary().data) - 1):
        row = tukey.summary().data[i + 1]
        posthoc_results.append({
            'Metabolite': metabolite,
            'Group1': f'Day {row[0]}',
            'Group2': f'Day {row[1]}',
            'Mean_Diff': row[2],
            'p_adj': row[3],
            'Lower_CI': row[4],
            'Upper_CI': row[5],
            'Significant': row[6]
        })

posthoc_df = pd.DataFrame(posthoc_results)
posthoc_df.to_csv('output/tukey_posthoc_results.csv', index=False)

# Print summary of significant pairwise comparisons
sig_comparisons = posthoc_df[posthoc_df['Significant'] == True]
print(f"\nSignificant pairwise comparisons: {len(sig_comparisons)}")

# Count significant comparisons per metabolite
sig_per_metabolite = sig_comparisons.groupby('Metabolite').size().sort_values(ascending=False)
print("\nMetabolites with most significant day-to-day changes:")
print(sig_per_metabolite.head(10).to_string())

# ============================================================================
# 4. Line Plot Visualization of Metabolites Over Time
# ============================================================================
print("\n" + "=" * 60)
print("4. Visualizing Metabolites Over Time")
print("=" * 60)

# Select top metabolites for visualization (top 20 by combined VIP and ANOVA significance)
top_n = 20
top_metabolites = anova_vip_df.head(top_n)['Metabolite'].tolist()

# Calculate mean and std for each metabolite at each time point
time_data = []
for metabolite in top_metabolites:
    for day in days_unique:
        day_mask = merged_data['Days'] == day
        values = merged_data.loc[day_mask, metabolite]
        time_data.append({
            'Metabolite': metabolite,
            'Day': day,
            'Mean': values.mean(),
            'Std': values.std(),
            'SEM': values.std() / np.sqrt(len(values))
        })

time_df = pd.DataFrame(time_data)

# Normalize data for better visualization (z-score per metabolite)
normalized_time_data = []
for metabolite in top_metabolites:
    met_data = time_df[time_df['Metabolite'] == metabolite].copy()
    mean_all = met_data['Mean'].mean()
    std_all = met_data['Mean'].std()
    if std_all > 0:
        met_data['Normalized_Mean'] = (met_data['Mean'] - mean_all) / std_all
    else:
        met_data['Normalized_Mean'] = 0
    normalized_time_data.append(met_data)

normalized_df = pd.concat(normalized_time_data, ignore_index=True)

# Create line plot with all metabolites
fig, ax = plt.subplots(figsize=(14, 10))

# Generate distinct colors
cmap = plt.colormaps['tab20']
colors = [cmap(i / top_n) for i in range(top_n)]

# Plot each metabolite
lines = []
for i, metabolite in enumerate(top_metabolites):
    met_data = normalized_df[normalized_df['Metabolite'] == metabolite].sort_values('Day')
    line, = ax.plot(met_data['Day'], met_data['Normalized_Mean'],
                    marker='o', linewidth=2, markersize=8,
                    color=colors[i], alpha=0.8)
    lines.append(line)

    # Annotate the last point with metabolite name
    last_point = met_data.iloc[-1]
    # Truncate long names for readability
    display_name = metabolite[:25] + '...' if len(metabolite) > 25 else metabolite
    ax.annotate(display_name,
                xy=(last_point['Day'], last_point['Normalized_Mean']),
                xytext=(5, 0), textcoords='offset points',
                fontsize=8, alpha=0.9,
                verticalalignment='center')

ax.set_xlabel('Days of Fermentation', fontsize=14)
ax.set_ylabel('Normalized Abundance (Z-score)', fontsize=14)
ax.set_title(f'Top {top_n} Discriminating Metabolites Over Fermentation Time', fontsize=16)
ax.set_xticks(days_unique)
ax.set_xticklabels([f'Day {d}' for d in days_unique])
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('output/metabolites_time_course.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Time course plot saved to: output/metabolites_time_course.png")

# ============================================================================
# Create a second visualization with faceted subplots and significance
# ============================================================================

def get_significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

n_cols = 4
n_rows = (top_n + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
axes = axes.flatten()

for i, metabolite in enumerate(top_metabolites):
    ax = axes[i]
    met_data = time_df[time_df['Metabolite'] == metabolite].sort_values('Day')

    ax.errorbar(met_data['Day'], met_data['Mean'], yerr=met_data['SEM'],
                marker='o', linewidth=2, markersize=6, capsize=3,
                color=colors[i])

    # Get ANOVA p-value for this metabolite
    p_adj = anova_vip_df[anova_vip_df['Metabolite'] == metabolite]['p_adjusted'].values[0]
    stars = get_significance_stars(p_adj)

    # Truncate name for title and add significance
    display_name = metabolite[:25] + '...' if len(metabolite) > 25 else metabolite
    ax.set_title(f'{display_name}\n(ANOVA: {stars}, p={p_adj:.2e})',
                 fontsize=9, fontweight='bold')
    ax.set_xlabel('Days', fontsize=9)
    ax.set_ylabel('Abundance', fontsize=9)
    ax.set_xticks(days_unique)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

# Hide unused subplots
for i in range(len(top_metabolites), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('output/metabolites_individual_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Individual metabolite plots saved to: output/metabolites_individual_plots.png")

# ============================================================================
# Save summary statistics
# ============================================================================
summary_stats = time_df.pivot_table(
    index='Metabolite',
    columns='Day',
    values=['Mean', 'Std'],
    aggfunc='first'
)
summary_stats.to_csv('output/metabolite_summary_stats.csv')
print(f"\nSummary statistics saved to: output/metabolite_summary_stats.csv")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
print("\nOutput files generated:")
print("  1. output/vip_scores.csv - VIP scores for all metabolites")
print("  2. output/plsda_scores_plot.png - PLS-DA scores visualization")
print("  3. output/anova_results.csv - ANOVA results with FDR correction")
print("  4. output/tukey_posthoc_results.csv - Tukey HSD pairwise comparisons")
print("  5. output/metabolites_time_course.png - Line plot of top metabolites")
print("  6. output/metabolites_individual_plots.png - Individual plots with significance")
print("  7. output/metabolite_summary_stats.csv - Summary statistics")
print("\nKey Findings:")
print(f"  - Total metabolites analyzed: {len(metabolite_columns)}")
print(f"  - Significant by VIP (VIP > 1): {len(significant_metabolites)}")
print(f"  - Significant by ANOVA (p_adj < 0.05): {anova_df['Significant'].sum()}")
print(f"  - Significant by ANOVA (p_adj < 0.001): {(anova_df['p_adjusted'] < 0.001).sum()}")
print(f"  - PLS-DA cross-validation accuracy: {cv_accuracy:.2%}")
print("\nSignificance notation: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")