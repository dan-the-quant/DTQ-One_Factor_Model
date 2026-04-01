# Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%%

#############################################
#         LOAD ALL BETAS STRUCTURES
#############################################

# Function for loading betas
def load_betas(path):
    df = pd.read_csv(path)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, dayfirst=True, format='mixed')
    return df

def load_beta_group(prefix, windows, freq):
    return {w: load_betas(f'Betas/{prefix}_{w}{freq}.csv') for w in windows}

windows_daily = [252, 504, 756, 1008, 1260]
windows_weekly = [52, 104, 156, 208, 260]
windows_monthly = [12, 24, 36, 48, 60]

ewma_daily_betas = load_beta_group('ewma_betas', windows_daily, 'd')
ewma_weekly_betas = load_beta_group('ewma_betas', windows_weekly, 'w')
ewma_monhly_betas = load_beta_group('ewma_betas', windows_monthly, 'm')

#%%

with open("Outputs/predicted_betas_daily_models.pkl", "rb") as f:
    predicted_betas_daily = pickle.load(f)

with open("Outputs/predicted_betas_weekly_models.pkl", "rb") as f:
    predicted_betas_weekly = pickle.load(f)

with open("Outputs/predicted_betas_monthly_models.pkl", "rb") as f:
    predicted_betas_monthly = pickle.load(f)

ewma_gls_daily_betas = {}
ewma_gls_weekly_betas = {}
ewma_gls_monthly_betas = {}

selected_daily = [
    'wls_ewma_standardized_alpha_252',
    'wls_ewma_standardized_alpha_504',
    'wls_ewma_standardized_alpha_756',
    'wls_ewma_standardized_alpha_1008',
    'wls_ewma_standardized_alpha_1260',
    ]

selected_weekly = [
    'wls_ewma_standardized_alpha_52',
    'wls_ewma_standardized_alpha_104',
    'wls_ewma_standardized_alpha_156',
    'wls_ewma_standardized_alpha_208',
    'wls_ewma_standardized_alpha_260',
    ]

selected_monthly = [
    'wls_ewma_standardized_alpha_12',
    'wls_ewma_standardized_alpha_24',
    'wls_ewma_standardized_alpha_36',
    'wls_ewma_standardized_alpha_48',
    'wls_ewma_standardized_alpha_60',
    ]

for w, k in zip(windows_daily, selected_daily):
    temp = predicted_betas_daily[k]
    temp.index = pd.to_datetime(temp.index)
    
    ewma_gls_daily_betas[w] = temp

for w, k in zip(windows_weekly, selected_weekly):
    temp = predicted_betas_weekly[k]
    temp.index = pd.to_datetime(temp.index)
    
    ewma_gls_weekly_betas[w] = temp

for w, k in zip(windows_monthly, selected_monthly):
    temp = predicted_betas_monthly[k]
    temp.index = pd.to_datetime(temp.index)
    
    ewma_gls_monthly_betas[w] = temp

#%%

#############################################
#   EXTRACT LAST AVAILABLE DATE EACH YEAR
#############################################

def get_last_date(df):
    """
    Returns a DataFrame containing cross-sectional betas
    at the last available date of each calendar year.
    """
    # Identify last available date per year
    # last_dates = df.groupby(df.index.year).apply(lambda x: x.index.max())
    last_dates = df.groupby(df.index.to_period('Q')).apply(lambda x: x.index.max())

    # Return cross-sections at those dates
    return df.loc[last_dates]


#############################################
#        PLOT CROSS-SECTIONAL DISTRIBUTIONS
#############################################

def plot_year_end_distributions(df, window, freq, title):

    # Get year-end cross-sections
    date_end_df = get_last_date(df)
    date_end_df = date_end_df.loc['2009':]

    data = []
    labels = []

    for date, row in date_end_df.iterrows():
        clean_row = row.dropna()
        if len(clean_row) > 0:
            data.append(clean_row.values)
            labels.append(date.strftime('%m-%Y'))

    if len(data) == 0:
        print(f"No valid data for window {window}")
        return

    plt.figure(figsize=(14,6))
    plt.boxplot(data, showfliers=True)

    plt.xticks(range(1, len(labels)+1), labels, rotation=90)

    plt.ylabel("Beta")
    plt.title(title)
    plt.grid(alpha=0.3)
    
    if freq == 'daily':
        plt.ylim(-2, 5)
    elif freq == 'weekly':
        plt.ylim(-6, 7)
    elif freq == 'monthly':
        plt.ylim(-10, 15)

    plt.tight_layout()
    plt.show()

#%%

#############################################
#            RUN FOR ALL WINDOWS
#############################################

for w, df in ewma_daily_betas.items():
    plot_year_end_distributions(df, w, 'daily', title=f'Cross-Sectional CAPM Betas Distribution ({w} - Days)')
    
for w, df in ewma_gls_daily_betas.items():
    plot_year_end_distributions(df, w, 'daily', title=f'Cross-Sectional Predicted Betas Distribution ({w} - Days)')

#%%

for w, df in ewma_weekly_betas.items():
    plot_year_end_distributions(df, w, 'weekly', title=f'Cross-Sectional CAPM Betas Distribution ({w} - Weeks)')

for w, df in ewma_gls_weekly_betas.items():
    plot_year_end_distributions(df, w, 'weekly', title=f'Cross-Sectional Predicted Betas Distribution ({w} - Weeks)')
    
#%%

for w, df in ewma_monhly_betas.items():
    plot_year_end_distributions(df, w, 'monthly')
    
for w, df in ewma_gls_monthly_betas.items():
    plot_year_end_distributions(df, w, 'monthly')
    
#%%

def compare_beta_distributions(df1, df2, window, freq, title1, title2):
    # Función auxiliar para procesar los datos de cada dataframe
    def get_plot_data(df):
        date_end_df = get_last_date(df).loc['2009':]
        data, labels = [], []
        for date, row in date_end_df.iterrows():
            clean_row = row.dropna()
            if len(clean_row) > 0:
                data.append(clean_row.values)
                labels.append(date.strftime('%m-%Y'))
        return data, labels

    data1, labels1 = get_plot_data(df1)
    data2, labels2 = get_plot_data(df2)

    # Creamos una figura con 2 subplots (1 fila, 2 columnas)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharey=True)

    # Configuración de límites según frecuencia
    y_limits = {'daily': (-2, 5), 'weekly': (-6, 7.5), 'monthly': (-10, 15)}
    curr_lims = y_limits.get(freq, None)

    # Plot 1: CAPM Betas
    axes[0].boxplot(data1, showfliers=True)
    axes[0].set_title(f"{title1} (w={window})")
    axes[0].set_xticklabels(labels1, rotation=90)
    axes[0].set_ylabel("Beta")
    axes[0].grid(alpha=0.3)

    # Plot 2: Predicted Betas
    axes[1].boxplot(data2, showfliers=True)
    axes[1].set_title(f"{title2} (w={window})")
    axes[1].set_xticklabels(labels2, rotation=90)
    axes[1].grid(alpha=0.3)

    if curr_lims:
        axes[0].set_ylim(curr_lims)
        axes[1].set_ylim(curr_lims)

    plt.tight_layout()
    plt.show()

#%%

# Iteramos sobre las llaves de uno (asumiendo que ambos tienen las mismas ventanas 'w')
for w in ewma_weekly_betas.keys():
    compare_beta_distributions(
        ewma_weekly_betas[w], 
        ewma_gls_weekly_betas[w], 
        w, 
        'weekly',
        title1='CAPM Betas',
        title2='Predicted Betas'
    )