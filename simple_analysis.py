"""
Simple analysis using only Python standard library
"""
import csv
import statistics
from collections import defaultdict, Counter

def load_csv(filename):
    """Load CSV file and return data as list of dicts"""
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)

def analyze_data():
    # Load data
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    recycled = load_csv('data/AllData_PreEDM_Recycled_RowColIDs.csv')
    virgin = load_csv('data/AllData_PreEDM_Virgin_RowColIDs.csv')

    all_data = recycled + virgin

    print(f"\nRecycled data: {len(recycled)} rows")
    print(f"Virgin data: {len(virgin)} rows")
    print(f"Total data: {len(all_data)} rows")

    # ========================================================================
    # DEFECT ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("DEFECT RATE ANALYSIS")
    print("="*80)

    # Overall defect rates
    total_defects = sum(1 for row in all_data if row['Nonconformity'] == 'TRUE')
    overall_rate = (total_defects / len(all_data)) * 100

    print(f"\nOverall Statistics:")
    print(f"Total parts: {len(all_data)}")
    print(f"Total defects: {total_defects}")
    print(f"Overall defect rate: {overall_rate:.2f}%")

    # By powder type
    print("\n--- Defect Rate by Powder Type ---")
    for powder in ['Virgin', 'Recycled']:
        powder_data = [row for row in all_data if row['Powder'] == powder]
        powder_defects = sum(1 for row in powder_data if row['Nonconformity'] == 'TRUE')
        powder_rate = (powder_defects / len(powder_data)) * 100 if powder_data else 0
        print(f"{powder:10s}: {len(powder_data):4d} parts, {powder_defects:2d} defects, {powder_rate:5.2f}% defect rate")

    # By layout
    print("\n--- Defect Rate by Layout ---")
    layouts = set(row['Layout'] for row in all_data)
    for layout in sorted(layouts):
        layout_data = [row for row in all_data if row['Layout'] == layout]
        layout_defects = sum(1 for row in layout_data if row['Nonconformity'] == 'TRUE')
        layout_rate = (layout_defects / len(layout_data)) * 100 if layout_data else 0
        parts_per_plate = len(layout_data) / len(set(row['PlateID'] for row in layout_data))
        print(f"{layout:10s}: {len(layout_data):4d} parts, {layout_defects:2d} defects, {layout_rate:5.2f}% rate, ~{parts_per_plate:.0f} parts/plate")

    # By powder AND layout
    print("\n--- Defect Rate by Powder AND Layout ---")
    for powder in ['Virgin', 'Recycled']:
        for layout in sorted(layouts):
            combo_data = [row for row in all_data if row['Powder'] == powder and row['Layout'] == layout]
            if combo_data:
                combo_defects = sum(1 for row in combo_data if row['Nonconformity'] == 'TRUE')
                combo_rate = (combo_defects / len(combo_data)) * 100
                parts_per_plate = len(combo_data) / len(set(row['PlateID'] for row in combo_data))
                good_parts = parts_per_plate * (1 - combo_rate/100)
                print(f"{powder:10s} + {layout:10s}: {len(combo_data):4d} parts, {combo_defects:2d} defects, {combo_rate:5.2f}% rate, {good_parts:.1f} good parts/plate")

    # ========================================================================
    # POSITIONAL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("POSITIONAL EFFECTS ANALYSIS")
    print("="*80)

    # Edge vs center
    edge_data = [row for row in all_data if row['RowID'] in ['1', '11'] or row['ColID'] in ['1', '11']]
    center_data = [row for row in all_data if row['RowID'] not in ['1', '11'] and row['ColID'] not in ['1', '11']]

    edge_defects = sum(1 for row in edge_data if row['Nonconformity'] == 'TRUE')
    center_defects = sum(1 for row in center_data if row['Nonconformity'] == 'TRUE')

    edge_rate = (edge_defects / len(edge_data)) * 100 if edge_data else 0
    center_rate = (center_defects / len(center_data)) * 100 if center_data else 0

    print(f"\nEdge positions (Row/Col 1 or 11):")
    print(f"  Total parts: {len(edge_data)}, Defects: {edge_defects}, Rate: {edge_rate:.2f}%")
    print(f"\nCenter positions:")
    print(f"  Total parts: {len(center_data)}, Defects: {center_defects}, Rate: {center_rate:.2f}%")

    # List all defect positions
    print("\n--- Defect Locations ---")
    defects = [row for row in all_data if row['Nonconformity'] == 'TRUE']
    print(f"\nAll {len(defects)} defects:")
    position_counts = defaultdict(int)
    for defect in defects:
        pos = f"({defect['RowID']:>2s},{defect['ColID']:>2s})"
        position_counts[pos] += 1
        print(f"  {defect['Row']:10s} - Row {defect['RowID']:>2s}, Col {defect['ColID']:>2s}, {defect['Powder']:8s}, {defect['Layout']:10s}, Plate {defect['PlateID']}")

    # ========================================================================
    # DIMENSIONAL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("DIMENSIONAL MEASUREMENTS")
    print("="*80)

    measurement_cols = ['B3_DATUM_B_LOC', 'B3_REF_OD', 'C1_LOC_INSIDE_PLN',
                        'C4_LOC_TOP_PLN', 'B3_THICK1_WALL', 'B3_THICK2_WALL',
                        'B3_THICK3_WALL', 'B3_THICK4_WALL']

    conforming = [row for row in all_data if row['Nonconformity'] == 'FALSE']
    nonconforming = [row for row in all_data if row['Nonconformity'] == 'TRUE']

    print("\nComparing Conforming vs Nonconforming parts:")
    for col in measurement_cols:
        conf_values = [float(row[col]) for row in conforming]
        nonconf_values = [float(row[col]) for row in nonconforming if row[col]]

        conf_mean = statistics.mean(conf_values)
        conf_std = statistics.stdev(conf_values) if len(conf_values) > 1 else 0

        if nonconf_values:
            nonconf_mean = statistics.mean(nonconf_values)
            nonconf_std = statistics.stdev(nonconf_values) if len(nonconf_values) > 1 else 0
            diff = abs(conf_mean - nonconf_mean)
            diff_pct = (diff / conf_mean) * 100

            print(f"\n{col}:")
            print(f"  Conforming:    Mean={conf_mean:.6f}, Std={conf_std:.6f}")
            print(f"  Nonconforming: Mean={nonconf_mean:.6f}, Std={nonconf_std:.6f}")
            print(f"  Difference:    {diff:.6f} ({diff_pct:.2f}%)")

    # ========================================================================
    # PLATE ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PLATE-TO-PLATE VARIATION")
    print("="*80)

    plates = set(row['PlateID'] for row in all_data)
    plate_stats = []

    for plate in sorted(plates):
        plate_data = [row for row in all_data if row['PlateID'] == plate]
        plate_defects = sum(1 for row in plate_data if row['Nonconformity'] == 'TRUE')
        plate_rate = (plate_defects / len(plate_data)) * 100 if plate_data else 0
        powder = plate_data[0]['Powder'] if plate_data else 'N/A'
        layout = plate_data[0]['Layout'] if plate_data else 'N/A'

        plate_stats.append((plate, len(plate_data), plate_defects, plate_rate, powder, layout))

    # Sort by defect rate
    plate_stats.sort(key=lambda x: x[3], reverse=True)

    print("\nPlates with defects:")
    for plate, parts, defects, rate, powder, layout in plate_stats:
        if defects > 0:
            print(f"Plate {plate}: {parts:3d} parts, {defects} defects, {rate:5.2f}% rate - {powder:8s}, {layout:10s}")

    print("\nPlates with NO defects:")
    no_defect_count = 0
    for plate, parts, defects, rate, powder, layout in plate_stats:
        if defects == 0:
            no_defect_count += 1
            if no_defect_count <= 10:  # Show first 10
                print(f"Plate {plate}: {parts:3d} parts, {defects} defects, {rate:5.2f}% rate - {powder:8s}, {layout:10s}")
    if no_defect_count > 10:
        print(f"... and {no_defect_count - 10} more plates with 0 defects")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)

    print("\n1. POWDER TYPE IMPACT:")
    print("   - Virgin powder shows LOWER defect rate (0.64% vs 1.27%)")
    print("   - Difference: ~2x improvement with virgin powder")

    print("\n2. LAYOUT & THROUGHPUT TRADEOFF:")
    print("   - 6X6: ~36 parts/plate (lower throughput)")
    print("   - 6X6TA: ~36 parts/plate (with thermal assistance)")
    print("   - 11X11TA: ~113 parts/plate (higher throughput, 3x more parts)")
    print("   - Higher throughput layouts may have higher defect rates")

    print("\n3. POSITIONAL EFFECTS:")
    if edge_rate > center_rate:
        print(f"   - Edge positions show HIGHER defect rate ({edge_rate:.2f}% vs {center_rate:.2f}%)")
    else:
        print(f"   - No strong edge effect detected")

    print("\n4. SAMPLE SIZE CONSIDERATIONS:")
    print(f"   - Total observations: {len(all_data)}")
    print(f"   - Total defects: {total_defects} (low event rate)")
    print(f"   - This low defect rate requires careful statistical approach")

    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_data()
