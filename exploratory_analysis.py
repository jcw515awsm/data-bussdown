"""
Comprehensive Exploratory Data Analysis for Additive Manufacturing Data
Goal: Determine optimal combination of plate layout, powder type, location, and heat sinks
      to minimize defective parts while maximizing throughput
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load data
print("="*80)
print("LOADING DATA")
print("="*80)

recycled = pd.read_csv('data/AllData_PreEDM_Recycled_RowColIDs.csv', encoding='utf-8-sig')
virgin = pd.read_csv('data/AllData_PreEDM_Virgin_RowColIDs.csv', encoding='utf-8-sig')

print(f"\nRecycled data shape: {recycled.shape}")
print(f"Virgin data shape: {virgin.shape}")

# Combine datasets
all_data = pd.concat([recycled, virgin], ignore_index=True)
print(f"Combined data shape: {all_data.shape}")

# Basic info
print("\n" + "="*80)
print("DATA STRUCTURE")
print("="*80)
print("\nColumn names:")
print(all_data.columns.tolist())
print("\nData types:")
print(all_data.dtypes)
print("\nFirst few rows:")
print(all_data.head())

# ============================================================================
# DEFECT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DEFECT RATE ANALYSIS")
print("="*80)

# Overall defect rates
total_parts = len(all_data)
total_defects = all_data['Nonconformity'].sum()
overall_defect_rate = total_defects / total_parts * 100

print(f"\nOverall Statistics:")
print(f"Total parts: {total_parts}")
print(f"Total defects: {total_defects}")
print(f"Overall defect rate: {overall_defect_rate:.2f}%")

# Defect rate by powder type
print("\n--- Defect Rate by Powder Type ---")
powder_defects = all_data.groupby('Powder').agg({
    'Nonconformity': ['count', 'sum', lambda x: (x.sum()/len(x)*100)]
}).round(2)
powder_defects.columns = ['Total_Parts', 'Defects', 'Defect_Rate_%']
print(powder_defects)

# Statistical test: Chi-square for powder type vs defects
powder_contingency = pd.crosstab(all_data['Powder'], all_data['Nonconformity'])
chi2, p_value, dof, expected = chi2_contingency(powder_contingency)
print(f"\nChi-square test (Powder vs Defects):")
print(f"Chi2 = {chi2:.4f}, p-value = {p_value:.4f}")
print(f"Statistically significant: {'YES' if p_value < 0.05 else 'NO'}")

# Defect rate by layout
print("\n--- Defect Rate by Layout ---")
layout_defects = all_data.groupby('Layout').agg({
    'Nonconformity': ['count', 'sum', lambda x: (x.sum()/len(x)*100)]
}).round(2)
layout_defects.columns = ['Total_Parts', 'Defects', 'Defect_Rate_%']
print(layout_defects)

# Defect rate by powder AND layout
print("\n--- Defect Rate by Powder Type AND Layout ---")
powder_layout_defects = all_data.groupby(['Powder', 'Layout']).agg({
    'Nonconformity': ['count', 'sum', lambda x: (x.sum()/len(x)*100)]
}).round(2)
powder_layout_defects.columns = ['Total_Parts', 'Defects', 'Defect_Rate_%']
print(powder_layout_defects)

# ============================================================================
# THROUGHPUT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("THROUGHPUT ANALYSIS")
print("="*80)

# Calculate parts per plate for each layout
layout_throughput = all_data.groupby(['Layout', 'PlateID']).size().reset_index(name='Parts_Per_Plate')
throughput_stats = layout_throughput.groupby('Layout')['Parts_Per_Plate'].agg(['mean', 'min', 'max'])
print("\nParts per plate by layout:")
print(throughput_stats)

# Throughput efficiency = (1 - defect_rate) * parts_per_plate
print("\n--- Throughput Efficiency (Good Parts per Plate) ---")
efficiency_data = []
for (powder, layout), group in all_data.groupby(['Powder', 'Layout']):
    total_parts = len(group)
    defect_rate = group['Nonconformity'].sum() / total_parts
    # Get typical parts per plate
    parts_per_plate = group.groupby('PlateID').size().mean()
    good_parts_per_plate = parts_per_plate * (1 - defect_rate)

    efficiency_data.append({
        'Powder': powder,
        'Layout': layout,
        'Parts_Per_Plate': parts_per_plate,
        'Defect_Rate_%': defect_rate * 100,
        'Good_Parts_Per_Plate': good_parts_per_plate,
        'Efficiency_%': (1 - defect_rate) * 100
    })

efficiency_df = pd.DataFrame(efficiency_data).round(2)
efficiency_df = efficiency_df.sort_values('Good_Parts_Per_Plate', ascending=False)
print(efficiency_df)

# ============================================================================
# POSITIONAL EFFECTS (Location on Plate)
# ============================================================================
print("\n" + "="*80)
print("POSITIONAL EFFECTS ANALYSIS")
print("="*80)

# Defects by position
print("\n--- Defects by Row Position ---")
row_defects = all_data.groupby('RowID').agg({
    'Nonconformity': ['count', 'sum', lambda x: (x.sum()/len(x)*100)]
}).round(2)
row_defects.columns = ['Total_Parts', 'Defects', 'Defect_Rate_%']
print(row_defects)

print("\n--- Defects by Column Position ---")
col_defects = all_data.groupby('ColID').agg({
    'Nonconformity': ['count', 'sum', lambda x: (x.sum()/len(x)*100)]
}).round(2)
col_defects.columns = ['Total_Parts', 'Defects', 'Defect_Rate_%']
print(col_defects)

# Check edge vs center effects
# Define edge as row/col 1 or 11, center as middle positions
all_data['Is_Edge'] = ((all_data['RowID'] == 1) | (all_data['RowID'] == 11) |
                        (all_data['ColID'] == 1) | (all_data['ColID'] == 11))

print("\n--- Edge vs Center Defect Rates ---")
edge_defects = all_data.groupby('Is_Edge').agg({
    'Nonconformity': ['count', 'sum', lambda x: (x.sum()/len(x)*100)]
}).round(2)
edge_defects.columns = ['Total_Parts', 'Defects', 'Defect_Rate_%']
edge_defects.index = ['Center', 'Edge']
print(edge_defects)

# Statistical test
edge_contingency = pd.crosstab(all_data['Is_Edge'], all_data['Nonconformity'])
chi2, p_value, dof, expected = chi2_contingency(edge_contingency)
print(f"\nChi-square test (Edge vs Center):")
print(f"Chi2 = {chi2:.4f}, p-value = {p_value:.4f}")

# ============================================================================
# DIMENSIONAL MEASUREMENTS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DIMENSIONAL MEASUREMENTS ANALYSIS")
print("="*80)

measurement_cols = ['B3_DATUM_B_LOC', 'B3_REF_OD', 'C1_LOC_INSIDE_PLN',
                     'C4_LOC_TOP_PLN', 'B3_THICK1_WALL', 'B3_THICK2_WALL',
                     'B3_THICK3_WALL', 'B3_THICK4_WALL']

# Compare measurements between conforming and nonconforming parts
print("\n--- Dimensional Statistics: Conforming vs Nonconforming ---")
for col in measurement_cols:
    conforming = all_data[all_data['Nonconformity'] == False][col]
    nonconforming = all_data[all_data['Nonconformity'] == True][col]

    # Mann-Whitney U test (non-parametric)
    statistic, p_value = mannwhitneyu(conforming, nonconforming, alternative='two-sided')

    print(f"\n{col}:")
    print(f"  Conforming    - Mean: {conforming.mean():.4f}, Std: {conforming.std():.4f}")
    print(f"  Nonconforming - Mean: {nonconforming.mean():.4f}, Std: {nonconforming.std():.4f}")
    print(f"  Difference: {abs(conforming.mean() - nonconforming.mean()):.4f}")
    print(f"  Mann-Whitney p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

# ============================================================================
# PLATE-TO-PLATE VARIATION
# ============================================================================
print("\n" + "="*80)
print("PLATE-TO-PLATE VARIATION")
print("="*80)

plate_stats = all_data.groupby('PlateID').agg({
    'Nonconformity': ['count', 'sum', lambda x: (x.sum()/len(x)*100)],
    'Powder': 'first',
    'Layout': 'first'
}).round(2)
plate_stats.columns = ['Total_Parts', 'Defects', 'Defect_Rate_%', 'Powder', 'Layout']
print("\nDefect rates by plate:")
print(plate_stats.sort_values('Defect_Rate_%', ascending=False))

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Create numeric encoding for categorical variables
all_data_encoded = all_data.copy()
all_data_encoded['Powder_Binary'] = (all_data_encoded['Powder'] == 'Recycled').astype(int)
all_data_encoded['Layout_6X6'] = (all_data_encoded['Layout'] == '6X6').astype(int)
all_data_encoded['Layout_6X6TA'] = (all_data_encoded['Layout'] == '6X6TA').astype(int)
all_data_encoded['Layout_11X11TA'] = (all_data_encoded['Layout'] == '11X11TA').astype(int)
all_data_encoded['Defect'] = all_data_encoded['Nonconformity'].astype(int)

# Correlation with defects
corr_cols = ['Defect', 'Powder_Binary', 'Layout_6X6', 'Layout_6X6TA',
             'Layout_11X11TA', 'RowID', 'ColID', 'Is_Edge'] + measurement_cols
correlations = all_data_encoded[corr_cols].corr()['Defect'].sort_values(ascending=False)

print("\nCorrelations with Defect (sorted by absolute value):")
print(correlations.sort_values(key=abs, ascending=False))

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("KEY FINDINGS SUMMARY")
print("="*80)

print("\n1. POWDER TYPE:")
print(f"   - Virgin powder: {(all_data[all_data['Powder']=='Virgin']['Nonconformity'].sum() / len(all_data[all_data['Powder']=='Virgin']) * 100):.2f}% defect rate")
print(f"   - Recycled powder: {(all_data[all_data['Powder']=='Recycled']['Nonconformity'].sum() / len(all_data[all_data['Powder']=='Recycled']) * 100):.2f}% defect rate")

print("\n2. LAYOUT (Throughput vs Quality):")
for layout in all_data['Layout'].unique():
    layout_data = all_data[all_data['Layout'] == layout]
    defect_rate = layout_data['Nonconformity'].sum() / len(layout_data) * 100
    parts_per_plate = layout_data.groupby('PlateID').size().mean()
    print(f"   - {layout}: {parts_per_plate:.0f} parts/plate, {defect_rate:.2f}% defect rate")

print("\n3. POSITIONAL EFFECTS:")
edge_rate = all_data[all_data['Is_Edge']]['Nonconformity'].sum() / len(all_data[all_data['Is_Edge']]) * 100
center_rate = all_data[~all_data['Is_Edge']]['Nonconformity'].sum() / len(all_data[~all_data['Is_Edge']]) * 100
print(f"   - Edge locations: {edge_rate:.2f}% defect rate")
print(f"   - Center locations: {center_rate:.2f}% defect rate")

print("\n4. SAMPLE SIZE:")
print(f"   - Total observations: {len(all_data)}")
print(f"   - Total defects: {all_data['Nonconformity'].sum()}")
print(f"   - Defect rate: {(all_data['Nonconformity'].sum() / len(all_data) * 100):.2f}%")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
