# Paso 0: Instalar dependencias
!pip install xgboost shap openpyxl

# Paso 1: Importar y cargar datos principales
import pandas as pd, numpy as np
import xgboost as xgb
import shap, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from google.colab import files

# Carga el histórico
uploaded = files.upload()
fn = next(iter(uploaded))
df_all = pd.read_csv(fn) if fn.lower().endswith('.csv') else pd.read_excel(fn, engine='openpyxl')
df_all['fecha'] = pd.to_datetime(df_all['fecha'], format='%Y-%m')
df_all = df_all.drop_duplicates(['cooler_id', 'fecha']).reset_index(drop=True)

# Paso 2: Crear variables derivadas e indicadores
umbral_power = df_all['power'].mean()
df_all['c_compresor'] = (df_all['compressor'] > df_all['compressor'].quantile(0.75)).astype(int)
df_all['c_energia'] = (df_all['power'] > umbral_power).astype(int)
df_all['c_flujo'] = (df_all['door_opens'] < df_all['door_opens'].quantile(0.25)).astype(int)
df_all['c_sellos'] = (df_all['voltage_dif'] > df_all['voltage_dif'].quantile(0.75)).astype(int)
df_all['c_termostato'] = (df_all['temperature'] > df_all['temperature'].quantile(0.75)).astype(int)

df_all = df_all.sort_values(['cooler_id', 'fecha'])
for lag in [1, 2, 3]:
    df_all[f'temp_lag{lag}'] = (df_all.groupby('cooler_id')['temperature']
                                 .shift(lag).fillna(method='bfill').fillna(0))
    df_all[f'power_lag{lag}'] = (df_all.groupby('cooler_id')['power']
                                  .shift(lag).fillna(method='bfill').fillna(0))

df_all['fallo_bin'] = df_all['Estado'].map({0: 0, 1: 1})

# Paso 3: División por IDs y entrenamiento XGBoost
X = df_all.drop(['Estado', 'fallo_bin', 'cooler_id', 'fecha'], axis=1)
y = df_all['fallo_bin']

train_ids, test_ids = train_test_split(df_all['cooler_id'].unique(),
                                       test_size=0.3, random_state=42)
mask_train = df_all['cooler_id'].isin(train_ids)

dtrain = xgb.DMatrix(X[mask_train], label=y[mask_train])
dtest = xgb.DMatrix(X[~mask_train], label=y[~mask_train])

bst = xgb.train({
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}, dtrain, num_boost_round=200, early_stopping_rounds=20,
   evals=[(dtrain, 'train'), (dtest, 'eval')], verbose_eval=False)

# Paso 4: Interpretación SHAP y métricas
expl = shap.TreeExplainer(bst)
sv = expl.shap_values(X[~mask_train])

plt.figure(figsize=(8, 5))
shap.summary_plot(sv, X[~mask_train], show=False)
plt.tight_layout()
plt.show()

probs = bst.predict(dtest)
roc = roc_auc_score(y[~mask_train], probs)
prec, rec, _ = precision_recall_curve(y[~mask_train], probs)
aucpr = auc(rec, prec)
print(f"ROC AUC: {roc:.3f} — AUC‑PR: {aucpr:.3f}")

# Paso 5: Asignar semáforo sobre todo el dataset
df_all['prob_fallo_all'] = bst.predict(xgb.DMatrix(X))
yellow_t, red_t = 0.08, 0.25
df_all['semaforo_all'] = df_all['prob_fallo_all'].apply(
    lambda p: "🔴 Rojo" if p >= red_t else
              ("🟠 Amarillo" if p >= yellow_t else "🟢 Verde")
)

# Paso 6: Carga lista de warnings
uploaded2 = files.upload()
fn2 = next(iter(uploaded2))
if fn2.lower().endswith(('.xls', '.xlsx')):
    df_warn = pd.read_excel(fn2, engine='openpyxl')
else:
    df_warn = pd.read_csv(fn2)
warn_ids = set(df_warn['cooler_id'].astype(str).unique())

# Paso 7: Filtrar coolers y quedar solo con la última observación
df_last = (df_all.sort_values(['cooler_id', 'fecha'])
                   .groupby('cooler_id').tail(1).reset_index(drop=True))
df_filtered = df_last[~df_last['cooler_id'].astype(str).isin(warn_ids)].reset_index(drop=True)

print(f"Coolers antes: {len(df_last)} — después de filtrar warning: {len(df_filtered)}")

# Paso 8: Cargar relaciones de cliente
print("Sube el archivo con la relación 'cooler_id - customer_id'")
uploaded = files.upload()
df_rel1 = pd.read_csv(next(iter(uploaded)))

print("Sube el archivo con la relación 'customer_id - tipo de cliente'")
uploaded = files.upload()
df_rel2 = pd.read_excel(next(iter(uploaded)))

# Paso 9: Unir relaciones y calcular prioridad
df_merged = df_filtered.merge(df_rel1[['cooler_id', 'customer_id']],
                              on='cooler_id', how='left')
df_merged = df_merged.merge(df_rel2[['customer_id', 'categoria']],
                            on='customer_id', how='left')

pesos = {'Grande': 3, 'Mediano': 2, 'Pequeño': 1}
df_merged['peso'] = df_merged['categoria'].map(pesos).fillna(0)
df_merged['prioridad'] = df_merged['prob_fallo_all'] * df_merged['peso']
df_merged['variable_mas_relevante'] = shap_mean.iloc[0]['feature']

# Paso 10: Preparar y exportar resultados
df_final = df_merged[[
    'cooler_id', 'customer_id', 'variable_mas_relevante',
    'prob_fallo_all', 'Estado', 'semaforo_all', 'categoria',
    'prioridad'
]].copy()

df_final.columns = [
    'cooler_id', 'customer_id', 'variable_mas_relevante',
    'probability', 'fallo', 'semaforo', 'tipo_cliente', 'prioridad'
]

df_final = df_final.sort_values(by='prioridad', ascending=False)

output_path = 'resultados_prioridad.xlsx'
df_final.to_excel(output_path, index=False)
files.download(output_path)
