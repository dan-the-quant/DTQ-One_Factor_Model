import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import matplotlib.pyplot as plt

#%%

windows_d = [252, 504, 756, 1008, 1260]
windows_w = [52, 104, 156, 208, 260]
windows_m = [12, 24, 36, 48, 60]

#%%

ewma_betas_d = pd.DataFrame()

for d in windows_d:
    
    testing_betas = pd.read_csv(rf'Betas\ewma_betas_{d}d.csv', index_col=0)
    testing_betas = testing_betas.dropna(axis=1, how='any')
    testing_betas = testing_betas.loc['2009':]
    
    ewma_betas_d[f'betas_{d}'] = testing_betas.mean(axis=1)

ewma_betas_d.index = pd.to_datetime(ewma_betas_d.index)

#%%

ewma_betas_w = pd.DataFrame()

for w in windows_w:
    
    testing_betas = pd.read_csv(rf'Betas\ewma_betas_{w}w.csv', index_col=0)
    testing_betas = testing_betas.dropna(axis=1, how='any')
    testing_betas = testing_betas.loc['2009':]
    
    ewma_betas_w[f'betas_{w}'] = testing_betas.mean(axis=1)

ewma_betas_w.index = pd.to_datetime(ewma_betas_w.index)

#%%

ewma_betas_m = pd.DataFrame()

for m in windows_m:
    
    testing_betas = pd.read_csv(rf'Betas\ewma_betas_{m}m.csv', index_col=0)
    testing_betas = testing_betas.dropna(axis=1, how='any')
    testing_betas = testing_betas.loc['2009':]
    
    ewma_betas_m[f'betas_{m}'] = testing_betas.mean(axis=1)

ewma_betas_m.index = pd.to_datetime(ewma_betas_m.index)

#%%

# Create the Plot
plt.figure(figsize=(10, 6))
plt.plot(ewma_betas_d['betas_252'], label='Daily 1y Betas', alpha=1)
plt.plot(ewma_betas_w['betas_52'], label='Weekly 1y Betas', alpha=1)
plt.plot(ewma_betas_m['betas_12'], label='Monthly 1y Betas', alpha=1)

# Config
plt.title('Betas Time Series')
plt.xlabel('Time')
plt.ylabel('Betas')
plt.legend()
plt.ylim(0.6, 1.5)

# Show
plt.grid()

plt.show()

#%%

# Create the Plot
plt.figure(figsize=(10, 6))
plt.plot(ewma_betas_d['betas_504'], label='Daily 2y Betas', alpha=1)
plt.plot(ewma_betas_w['betas_104'], label='Weekly 2y Betas', alpha=1)
plt.plot(ewma_betas_m['betas_24'], label='Monthly 2y Betas', alpha=1)

# Config
plt.title('Betas Time Series')
plt.xlabel('Time')
plt.ylabel('Betas')
plt.legend()
plt.ylim(0.6, 1.5)

# Show
plt.grid()

plt.show()

#%%

# Create the Plot
plt.figure(figsize=(10, 6))
plt.plot(ewma_betas_d['betas_756'], label='Daily 3y Betas', alpha=1)
plt.plot(ewma_betas_w['betas_156'], label='Weekly 3y Betas', alpha=1)
plt.plot(ewma_betas_m['betas_36'], label='Monthly 3y Betas', alpha=1)

# Config
plt.title('Betas Time Series')
plt.xlabel('Time')
plt.ylabel('Betas')
plt.legend()
plt.ylim(0.6, 1.5)

# Show
plt.grid()

plt.show()

#%%

# Create the Plot
plt.figure(figsize=(10, 6))
plt.plot(ewma_betas_d['betas_1008'], label='Daily 4y Betas', alpha=1)
plt.plot(ewma_betas_w['betas_208'], label='Weekly 4y Betas', alpha=1)
plt.plot(ewma_betas_m['betas_48'], label='Monthly 4y Betas', alpha=1)

# Config
plt.title('Betas Time Series')
plt.xlabel('Time')
plt.ylabel('Betas')
plt.legend()
plt.ylim(0.6, 1.5)

# Show
plt.grid()

plt.show()

#%%

# Create the Plot
plt.figure(figsize=(10, 6))
plt.plot(ewma_betas_d['betas_1260'], label='Daily 5y Betas', alpha=1)
plt.plot(ewma_betas_w['betas_260'], label='Weekly 5y Betas', alpha=1)
plt.plot(ewma_betas_m['betas_60'], label='Monthly 5y Betas', alpha=1)

# Config
plt.title('Betas Time Series')
plt.xlabel('Time')
plt.ylabel('Betas')
plt.legend()
plt.ylim(0.6, 1.5)

# Show
plt.grid()

plt.show()

#%%

fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

panels = [
    ('betas_252','betas_52','betas_12','1-Year Betas'),
    ('betas_504','betas_104','betas_24','2-Year Betas'),
    ('betas_756','betas_156','betas_36','3-Year Betas'),
    ('betas_1008','betas_208','betas_48','4-Year Betas'),
    ('betas_1260','betas_260','betas_60','5-Year Betas'),
]

for ax, (d,w,m,title) in zip(axes, panels):
    ax.plot(ewma_betas_d[d], label='Daily')
    ax.plot(ewma_betas_w[w], label='Weekly')
    ax.plot(ewma_betas_m[m], label='Monthly')
    ax.set_title(title)
    ax.set_ylim(0.6, 1.5)   
    ax.legend()
    ax.grid()

fig.supxlabel('Time')
fig.supylabel('Beta')

plt.tight_layout()
plt.show()

