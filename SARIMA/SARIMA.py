import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys

warnings.filterwarnings('ignore')

IN_COLAB = 'google.colab' in sys.modules

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "statsmodels"])
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller

FILENAME = 'btc_dataset_com_sinais.csv'

try:
    df_raw = pd.read_csv(FILENAME, nrows=3)
    print(f"Colunas encontradas: {list(df_raw.columns)}")
except FileNotFoundError:
    print(f"Erro: '{FILENAME}' não encontrado. Coloque o arquivo em /content (Colab) ou na mesma pasta.")
    sys.exit(1)

date_col = None
for c in ['date','Date','DATE','timestamp','Timestamp','time','Time','data','Data','mes','Mes','month','Month']:
    if c in df_raw.columns:
        date_col = c
        break
if date_col is None:
    for c in df_raw.columns:
        try:
            pd.to_datetime(df_raw[c])
            date_col = c
            break
        except Exception:
            continue

if date_col:
    df = pd.read_csv(FILENAME, index_col=date_col, parse_dates=True)
    df.index = pd.to_datetime(df.index, infer_datetime_format=True, errors='coerce')
    df = df[df.index.notna()]
    df.sort_index(inplace=True)
    print(f"Coluna de data: '{date_col}'")
else:
    df = pd.read_csv(FILENAME)
    df.index = pd.date_range(start='2019-01-01', periods=len(df), freq='MS')
    print("Aviso: sem coluna de data. Usando índice mensal sequencial.")

if df.index.min().year < 2019:
    df = df[df.index >= '2019-01-01']

col = None
for c in ['weighted_price','close','Close','CLOSE','price','Price','preco','Preco',
          'adj_close','Adj Close','open','Open','btc_close','btc_price','BTC_close','BTC_Close']:
    if c in df.columns:
        col = c
        break
if col is None:
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        print("Erro: nenhuma coluna numérica encontrada.")
        sys.exit(1)
    col = num_cols[0]

print(f"Coluna de preço: '{col}' | Registros: {len(df)}")

df[col] = pd.to_numeric(df[col], errors='coerce')
df = df[df[col].notna() & (df[col] > 0)]

prices_raw = df[col].values
dates_raw  = df.index

if len(prices_raw) < 12:
    print(f"Erro: apenas {len(prices_raw)} registros. Mínimo necessário: 12.")
    sys.exit(1)

s_tmp = pd.Series(prices_raw, index=dates_raw)
freq  = pd.infer_freq(dates_raw)
print(f"Frequência detectada: {freq}")

if freq not in ['MS', 'M', 'ME', 'BMS', 'BM']:
    print("Reamostrando para mensal (último valor do mês)...")
    s_tmp = s_tmp.resample('MS').last().dropna()

prices_raw = s_tmp.values
dates_raw  = s_tmp.index
print(f"Meses disponíveis: {len(prices_raw)}")

log_raw = np.log(prices_raw)
s_log   = pd.Series(log_raw, index=dates_raw)

WIN = 7
roll_med = s_log.rolling(WIN, center=True, min_periods=3).median()
roll_iqr = s_log.rolling(WIN, center=True, min_periods=3).quantile(0.75) \
         - s_log.rolling(WIN, center=True, min_periods=3).quantile(0.25)
roll_iqr = roll_iqr.fillna(roll_iqr.median())

FACTOR = 3.0
is_outlier = (s_log > roll_med + FACTOR * roll_iqr) | \
             (s_log < roll_med - FACTOR * roll_iqr)

n_out = is_outlier.sum()
if n_out:
    print(f"Outliers detectados (janela rolante 3×IQR): {n_out} meses → interpolando...")
    s_log_clean = s_log.copy()
    s_log_clean[is_outlier] = np.nan
    s_log_clean = s_log_clean.interpolate(method='linear').bfill().ffill()
else:
    s_log_clean = s_log.copy()
    print("Nenhum outlier detectado.")

log_prices = s_log_clean.values
prices     = np.exp(log_prices)
dates      = dates_raw

train_size   = int(len(prices) * 0.8)
train_log    = log_prices[:train_size]
test_log     = log_prices[train_size:]
train_prices = prices[:train_size]
test_prices  = prices[train_size:]
train_dates  = dates[:train_size]
test_dates   = dates[train_size:]

print(f"\nTreino: {train_dates[0].strftime('%Y-%m')} → {train_dates[-1].strftime('%Y-%m')} ({len(train_prices)} meses)")
print(f"Teste:  {test_dates[0].strftime('%Y-%m')} → {test_dates[-1].strftime('%Y-%m')} ({len(test_prices)} meses)")

if len(test_prices) < 2:
    print("Erro: teste com menos de 2 pontos.")
    sys.exit(1)

log_margin     = np.log(4.0)
max_log_clip   = np.max(train_log) + log_margin
min_log_clip   = np.min(train_log) - log_margin
max_price_clip = train_prices.max() * 4
min_price_clip = train_prices.min() * 0.2

adf_res = adfuller(train_log, autolag='AIC')
d       = 0 if adf_res[1] <= 0.05 else 1
print(f"\nADF p-valor={adf_res[1]:.4f} → d={d}")

s_period = 12 if len(train_log) >= 24 else max(2, len(train_log) // 3)
print(f"\nGrid Search SARIMA (período sazonal={s_period})...")

configs = [
    ((1,d,1), (1,1,1,s_period)),
    ((2,d,1), (1,1,1,s_period)),
    ((1,d,2), (1,1,1,s_period)),
    ((2,d,2), (1,1,1,s_period)),
    ((3,d,1), (1,1,1,s_period)),
    ((1,d,3), (1,1,1,s_period)),
    ((1,d,1), (0,1,1,s_period)),
    ((2,d,1), (0,1,1,s_period)),
]

best_aic    = np.inf
best_model  = None
best_params = None

for order, seasonal in configs:
    try:
        fit = SARIMAX(train_log,
                      order=order,
                      seasonal_order=seasonal,
                      exog=None,
                      enforce_stationarity=False,
                      enforce_invertibility=False
                      ).fit(disp=False, maxiter=300)
        print(f"  SARIMA{order}x{seasonal} → AIC={fit.aic:.2f}")
        if fit.aic < best_aic:
            best_aic    = fit.aic
            best_model  = fit
            best_params = (order, seasonal)
    except Exception as e:
        print(f"  SARIMA{order}x{seasonal} → FALHOU ({e})")

if best_model is None:
    print("Aviso: todos falharam, usando fallback SARIMA(1,1,1)x(1,1,1,12).")
    try:
        best_model  = SARIMAX(train_log, order=(1,1,1),
                              seasonal_order=(1,1,1,s_period),
                              exog=None,
                              enforce_stationarity=False,
                              enforce_invertibility=False
                              ).fit(disp=False, maxiter=300)
        best_params = ((1,1,1), (1,1,1,s_period))
    except Exception as e:
        print(f"Erro fatal: {e}"); sys.exit(1)

order, seasonal = best_params
print(f"\nMelhor: SARIMA{order}x{seasonal} | AIC={best_aic:.2f}")

# --- INÍCIO DA CORREÇÃO DA LINHA DO GRÁFICO ---
in_pred = best_model.get_prediction(start=0, end=len(train_log)-1, dynamic=False)
p_log   = in_pred.predicted_mean
p_log   = p_log.values if hasattr(p_log, 'values') else np.array(p_log)
p_log   = np.clip(p_log, min_log_clip, max_log_clip)

fitted_prices = np.exp(p_log)
fitted_dates  = train_dates

# Filtro de Sanidade: Transforma anomalias (picos irreais do modelo) em 'NaN'.
# Se o preço ajustado for 1.5x maior que o teto real ou 0.5x menor que o piso, 
# o Matplotlib não vai desenhar esse pedaço quebrado da linha.
limit_high = train_prices.max() * 1.5
limit_low  = train_prices.min() * 0.5

invalid_mask = (fitted_prices > limit_high) | (fitted_prices < limit_low)
fitted_prices[invalid_mask] = np.nan
# --- FIM DA CORREÇÃO ---

fc     = best_model.get_forecast(steps=len(test_log))
fc_log = fc.predicted_mean
fc_log = fc_log.values if hasattr(fc_log, 'values') else np.array(fc_log)

ci = fc.conf_int()
if hasattr(ci, 'iloc'):
    lo_log = ci.iloc[:, 0].values
    hi_log = ci.iloc[:, 1].values
else:
    lo_log = ci[:, 0]
    hi_log = ci[:, 1]

fc_log = np.clip(fc_log, min_log_clip, max_log_clip)
lo_log = np.clip(lo_log, min_log_clip, max_log_clip)
hi_log = np.clip(hi_log, min_log_clip, max_log_clip)

predictions  = np.clip(np.exp(fc_log), min_price_clip, max_price_clip)
lower_bound  = np.clip(np.exp(lo_log), min_price_clip, max_price_clip)
upper_bound  = np.clip(np.exp(hi_log), min_price_clip, max_price_clip)

mae       = np.mean(np.abs(test_prices - predictions))
rmse      = np.sqrt(np.mean((test_prices - predictions)**2))
mape      = np.mean(np.abs((test_prices - predictions) / test_prices)) * 100
naive_mae = np.mean(np.abs(np.diff(train_prices))) if len(train_prices) > 1 else mae
mase      = mae / naive_mae if naive_mae else np.inf
dac       = np.mean(np.sign(np.diff(test_prices)) == np.sign(np.diff(predictions))) * 100

print("\n" + "="*70)
print("Métricas (Teste):")
print(f"  MAE  = ${mae:,.2f}   RMSE = ${rmse:,.2f}")
print(f"  MAPE = {mape:.2f}%   MASE = {mase:.3f}   DAC = {dac:.1f}%")
print("="*70 + "\n")

for s in ['seaborn-v0_8-whitegrid','seaborn-whitegrid','ggplot']:
    try: plt.style.use(s); break
    except: continue

fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(dates, prices,               'o-', label='Real',          color='#2E86AB', lw=2,   ms=4, alpha=0.85, zorder=4)
ax.plot(fitted_dates, fitted_prices, '-',  label='Ajuste Treino', color='#06A77D', lw=2.5,       alpha=0.90, zorder=3)
ax.plot(test_dates, predictions,     's-', label='Previsão',      color='#D62828', lw=2,   ms=5, alpha=0.95, zorder=5)
ax.fill_between(test_dates, lower_bound, upper_bound, alpha=0.12, color='#D62828', label='IC 95%')
ax.axvline(dates[train_size], color='gray', ls='--', lw=1.8, alpha=0.6, label='Início Teste')
ax.set_title(f'Bitcoin – SARIMA{order}×{seasonal}', fontsize=14, fontweight='bold')
ax.set_xlabel('Data'); ax.set_ylabel('Preço (USD)')
ax.legend(loc='best'); ax.grid(alpha=0.3)
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(test_dates, test_prices,  'o-', label='Real',     color='#2E86AB', lw=2.5, ms=7, zorder=4)
ax.plot(test_dates, predictions,  's-', label='Previsto', color='#D62828', lw=2.5, ms=6, alpha=0.9, zorder=5)
ax.fill_between(test_dates, lower_bound, upper_bound, alpha=0.12, color='#D62828', label='IC 95%')
ax.set_title(f'Zoom Teste | MASE={mase:.3f} | MAPE={mape:.1f}%', fontsize=13, fontweight='bold')
ax.set_xlabel('Data'); ax.set_ylabel('Preço (USD)')
ax.legend(); ax.grid(alpha=0.3)
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(test_prices, predictions, alpha=0.75, s=90, edgecolors='k', lw=1.0, zorder=4)
mn = min(test_prices.min(), predictions.min()) * 0.95
mx = max(test_prices.max(), predictions.max()) * 1.05
ax.plot([mn, mx], [mn, mx], 'r--', lw=2, alpha=0.7)
try:
    r2 = np.corrcoef(test_prices, predictions)[0,1]**2
except:
    r2 = 0.0
ax.set_title(f'Real vs. Previsto  (R²={r2:.3f})', fontsize=13, fontweight='bold')
ax.set_xlabel('Real (USD)'); ax.set_ylabel('Previsto (USD)')
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

print("Concluído com sucesso!")