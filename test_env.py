# test_env.py
import sys

print("Python version:", sys.version)
print()

# --- Pandas Test ---
try:
    import pandas as pd
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    print("Pandas version:", pd.__version__)
    print("DataFrame test passed:")
    print(df.describe(), "\n")
except Exception as e:
    print("Pandas test failed:", e, "\n")

# --- NumPy Test ---
try:
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    print("NumPy version:", np.__version__)
    print("Mean of array:", np.mean(arr), "\n")
except Exception as e:
    print("NumPy test failed:", e, "\n")

# --- SciPy Test ---
try:
    from scipy import stats
    data = [1, 2, 3, 4, 5]
    t_stat, p_val = stats.ttest_1samp(data, 3)
    print("SciPy version:", stats.__version__)
    print(f"T-test result: t={t_stat:.3f}, p={p_val:.3f}")
except Exception as e:
    print("SciPy test failed:", e, "\n")
