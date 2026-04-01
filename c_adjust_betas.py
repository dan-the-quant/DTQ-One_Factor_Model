import pandas as pd
from pathlib import Path


# ============================================
# LOAD HELPERS
# ============================================
def load_betas(path):
    df = pd.read_csv(path)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, dayfirst=True, format='mixed')
    return df


# ============================================
# SHRINKAGE
# ============================================
def shrink_to_target(df, weight=2 / 3, target=1):
    return weight * df + (1 - weight) * target


# ============================================
# STANDARDIZATION
# ============================================
def beta_standardization(df: pd.DataFrame) -> pd.DataFrame:
    std = df.std(axis=1)
    z = (df.sub(1, axis=0)).div(std, axis=0)
    return z


# ============================================
# CONFIG
# ============================================
FREQS = {
    "daily": [252, 504, 756, 1008, 1260],
    "weekly": [52, 104, 156, 208, 260],
    "monthly": [12, 24, 36, 48, 60],
}

INPUT = Path("Betas")  # donde guardó el pipeline original
OUTPUT = Path("Betas")  # puedes separarlo si quieres

#%% RUN ADJUSTMENTS

for freq, windows in FREQS.items():

    for w in windows:
        # ------------------------
        # Load SMA / EWMA betas
        # ------------------------
        sma_path = INPUT / f"sma_betas_{w}{freq[0]}.csv"
        ewma_path = INPUT / f"ewma_betas_{w}{freq[0]}.csv"

        sma = load_betas(sma_path)
        ewma = load_betas(ewma_path)

        # ------------------------
        # Compute adjustments
        # ------------------------
        sma_shrunk = shrink_to_target(sma)
        ewma_shrunk = shrink_to_target(ewma)

        sma_std = beta_standardization(sma)
        ewma_std = beta_standardization(ewma)

        # ------------------------
        # Save
        # ------------------------
        sma_shrunk.to_csv(OUTPUT / f"sma_shrunk_{w}{freq[0]}.csv")
        ewma_shrunk.to_csv(OUTPUT / f"ewma_shrunk_{w}{freq[0]}.csv")

        sma_std.to_csv(OUTPUT / f"sma_standardized_{w}{freq[0]}.csv")
        ewma_std.to_csv(OUTPUT / f"ewma_standardized_{w}{freq[0]}.csv")
