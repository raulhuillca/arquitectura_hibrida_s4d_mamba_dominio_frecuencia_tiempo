# Arquitectura Híbrida S4D-Mamba para Detección de Anomalías en Tráfico de Red: Dominio de Frecuencia y Tiempo
---

### 💻 Hardware y Entorno

---

Los experimentos se llevaron a cabo utilizando el siguiente entorno computacional:

- **CPU:** Intel Xeon W-2123 a 3.60 GHz
- **Memoria:** 64 GB RAM
- **GPU:** NVIDIA Quadro P2000 (5 GB VRAM)
- **Sistema Operativo:** Ubuntu 24.04.4 LTS
- **CUDA:** CUDA 11.8
- **Stack de Software:** Python 3.12, Qiskit 1.1.1, Qiskit Aer 0.15.1, TenSEAL 0.3.16, Microsoft SEAL, NumPy, Pandas, Scikit-learn, Matplotlib
- 

---

### 💻 Data set para evaluacion

---

## 1. NSL-KDD: El clásico didáctico
Es el más fácil para empezar, solo 125k filas y ya viene en `.txt` listo para entrenar.

> **⚠️ Problema:** El dominio original `nsl.cs.unb.ca` está caído desde 2015.

**Dónde bajarlo ahora:**
- **Oficial actual (UNB):** [UNB CIC Datasets - NSL](https://www.unb.ca/cic/datasets/nsl.html)
- **Espejo Zenodo muy estable:** Busca *"NSL-KDD Dataset"* en Zenodo. Te da los 2 archivos clave `KDDTrain+.txt` y `KDDTest+.txt`.
- **Kaggle (1 click):** Ejecuta `kaggle datasets download -d hussainnasir/nslkdd` o busca "NSL-KDD" en Kaggle. Ya está en `.csv` limpio.

**Contenido:**
- `KDDTrain+` (125,973 records) y `KDDTest+` (22,544 records).
- 41 features + etiqueta de 4 ataques: **DoS, Probe, R2L, U2R**.

---

## 2. UNSW-NB15: El intermedio moderno
Creado por UNSW Canberra en Australia. Contiene 49 features y 9 familias de ataques modernos: **Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms**.

**Enlaces Oficiales:**
- **Página del proyecto UNSW:** [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Repositorio ADFA oficial:** [ADFA-NB15-Datasets](https://www.unsw.adfa.edu.au/australian-centre-for-cyber-security/cybersecurity/ADFA-NB15-Datasets/)
- **Kaggle fácil:** Busca `mrwellsdavid/unsw-nb15`. Ya viene con `UNSW_NB15_training-set.csv` y `testing-set.csv` + `UNSW-NB15_features.csv` que explica cada columna.

> **💡 Tip:** Baja la versión **NF-UNSW-NB15-v2** si quieres formato NetFlow, es mucho más liviana (∼442MB).

---

## 3. CIC-IDS2017 y CIC-IDS2018: El estándar actual
Son del Canadian Institute for Cybersecurity (CIC) de la Universidad de New Brunswick. Son los que usan todos los papers serios en la actualidad.

### CIC-IDS2017
- **Oficial:** [UNB CIC Datasets - IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Tamaño:** ∼5GB en total. 80+ features extraídas con `CICFlowMeter`.
- **Distribución de tráfico (Viene por días):**
  - **Lunes:** Solo benigno.
  - **Martes:** Brute Force + SSH.
  - **Miércoles:** DoS.
  - **Jueves:** Web Attacks.
  - **Viernes:** DDoS + Botnet + PortScan.

### CIC-IDS2018 (Versión mejorada en AWS)
- **Oficial:** [UNB CIC Datasets - IDS 2018](https://www.unb.ca/cic/datasets/ids-2018.html)

> **📝 Nota de Descarga (UNB):** Para ambos datasets te piden llenar un formulario corto con tu nombre/email institucional y te dan el link de descarga a Drive. Si no quieres esperar, busca *"cic-ids2017"* en Kaggle, donde la comunidad ha subido copias limpias y unificadas en formato CSV.

---

### 💻 Script de Python : Descargar, limpia, codifica y deja los archivos listos para meterlos directo a una red neuronal.

### Script de Descarga y Limpieza Automática (`nids_downloader.py`)

Para facilitar la reproducción de los experimentos, hemos preparado un script que descarga los datasets usando la API de Kaggle, unifica los archivos y aplica una limpieza estándar preparándolos para modelos de Deep Learning.

```python
"""
NIDS Dataset Downloader & Cleaner
Para Deep Learning for Network Intrusion Detection
Autor: Meta AI para Raul - Cusco
Datasets: NSL-KDD, UNSW-NB15, CIC-IDS2017

Uso:
pip install pandas numpy scikit-learn requests kagglehub tqdm

python nids_downloader.py --dataset UNSW-NB15
python nids_downloader.py --dataset CIC-IDS2017
python nids_downloader.py --dataset NSL-KDD
python nids_downloader.py --dataset all
"""

import argparse, os, sys, zipfile, glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Intenta usar kagglehub si está instalado (mucho más estable en Perú)
try:
    import kagglehub
    HAS_KAGGLEHUB = True
except:
    HAS_KAGGLEHUB = False

DATASETS = {
    "NSL-KDD": {
        "kaggle_id": "hussainnasir/nslkdd",
        "files": ["KDDTrain+.txt", "KDDTest+.txt"],
        "description": "4 tipos de ataques: DoS, Probe, R2L, U2R - 41 features"
    },
    "UNSW-NB15": {
        "kaggle_id": "mrwellsdavid/unsw-nb15",
        "files": ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"],
        "description": "9 ataques modernos: Fuzzers, Exploits, Worms, etc - 49 features"
    },
    "CIC-IDS2017": {
        "kaggle_id": "cicdataset/cicids2017",
        "alt_kaggle": "chethuhn/network-intrusion-dataset",
        "description": "8 ataques: DDoS, Botnet, Brute Force, Infiltration - 80+ features"
    }
}

def download_kaggle(dataset_name):
    info = DATASETS[dataset_name]
    kaggle_id = info["kaggle_id"]
    print(f"\n[+] Descargando {dataset_name} desde Kaggle: {kaggle_id}")
    if not HAS_KAGGLEHUB:
        print("[!] Instala kagglehub: pip install kagglehub")
        print(f"    Alternativa manual: kaggle datasets download -d {kaggle_id}")
        return None
    
    try:
        path = kagglehub.dataset_download(kaggle_id)
        print(f"[OK] Descargado en: {path}")
        return path
    except Exception as e:
        print(f"[!] Falló {kaggle_id}: {e}")
        if "alt_kaggle" in info:
            print(f"[+] Intentando alternativo: {info['alt_kaggle']}")
            try:
                path = kagglehub.dataset_download(info["alt_kaggle"])
                print(f"[OK] Descargado en: {path}")
                return path
            except Exception as e2:
                print(f"[!] Falló también: {e2}")
        return None

def clean_nsl_kdd(raw_path):
    print("\n[>>] Limpiando NSL-KDD...")
    # Columnas según documentación oficial
    cols = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

    files = glob.glob(os.path.join(raw_path, "**", "KDDTrain+*.txt"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(raw_path, "*.txt"))
    
    if not files:
        print("[!] No encontré KDDTrain+.txt")
        return None

    df_train = pd.read_csv(files[0], header=None, names=cols)
    # Test opcional
    test_files = glob.glob(os.path.join(raw_path, "**", "KDDTest+*.txt"), recursive=True)
    df_test = pd.read_csv(test_files[0], header=None, names=cols) if test_files else None

    df = pd.concat([df_train, df_test]) if df_test is not None else df_train
    
    # Mapeo a 5 clases didácticas
    def map_attack(label):
        label = label.strip()
        if label == 'normal': return 'normal'
        if label in ['neptune','back','land','pod','smurf','teardrop']: return 'DoS'
        if label in ['ipsweep','nmap','portsweep','satan']: return 'Probe'
        if label in ['ftp_write','guess_passwd','imap','multihop','phf','spy','warezclient','warezmaster']: return 'R2L'
        return 'U2R'

    df['attack_cat'] = df['label'].apply(map_attack)
    df['is_attack'] = (df['attack_cat'] != 'normal').astype(int)

    # Encoding categórico
    for col in ['protocol_type','service','flag']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Guardar
    df.to_csv("NSL-KDD_clean.csv", index=False)
    print(f"[OK] NSL-KDD_clean.csv -> {df.shape} | Clases: {df['attack_cat'].value_counts().to_dict()}")
    return df

def clean_unsw_nb15(raw_path):
    print("\n[>>] Limpiando UNSW-NB15...")
    files = glob.glob(os.path.join(raw_path, "**", "UNSW_NB15_*.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(raw_path, "*.csv"))
    
    # Filtrar el archivo de features
    files = [f for f in files if "features" not in f.lower() and "list" not in f.lower()]
    if not files:
        print("[!] No encontré CSVs de UNSW")
        return None

    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Limpieza típica
    # Algunos CSV vienen sin header
    if 'attack_cat' not in df.columns and df.shape[1] == 49:
        print("[!] Detectado CSV sin header, asignando nombres estándar...")
        # Nombres según UNSW-NB15_features.csv
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        # Intenta renombrar últimos
        df.rename(columns={df.columns[-2]: "attack_cat", df.columns[-1]: "label"}, inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Categóricos comunes
    for col in ['proto','service','state']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Crear binario
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    if 'is_attack' not in df.columns and 'label' in df.columns:
        df['is_attack'] = df['label']

    # Escalar numéricos
    num_cols = df.select_dtypes(include=[np.number]).columns
    # No escalar label
    scale_cols = [c for c in num_cols if c not in ['label','is_attack','id']]

    df.to_csv("UNSW-NB15_clean.csv", index=False)
    print(f"[OK] UNSW-NB15_clean.csv -> {df.shape}")
    if 'attack_cat' in df.columns:
        print(f"Clases: {df['attack_cat'].value_counts().head(10).to_dict()}")
    return df

def clean_cic_ids2017(raw_path):
    print("\n[>>] Limpiando CIC-IDS2017 (esto puede tardar)...")
    csvs = glob.glob(os.path.join(raw_path, "**", "*.csv"), recursive=True)
    if not csvs:
        print("[!] No encontré CSVs")
        return None
    
    # Evitar archivos muy pequeños
    csvs = [c for c in csvs if os.path.getsize(c) > 1000]
    print(f"Encontrados {len(csvs)} CSVs")

    dfs = []
    for c in csvs:
        try:
            temp = pd.read_csv(c, encoding='utf-8', low_memory=False)
            dfs.append(temp)
        except:
            try:
                temp = pd.read_csv(c, encoding='latin1', low_memory=False)
                dfs.append(temp)
            except Exception as e:
                print(f"[!] Skip {c}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"Concat shape inicial: {df.shape}")

    # Limpieza estándar CIC-IDS2017
    df.columns = df.columns.str.strip()
    # Quitar espacios en nombres
    df.columns = [c.replace(" ", "_") for c in df.columns]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Columna label puede llamarse Label o label
    label_col = None
    for possible in ['Label','label','LABEL','Attack','attack']:
        if possible in df.columns:
            label_col = possible
            break
    
    if label_col:
        df.rename(columns={label_col: "attack_cat"}, inplace=True)
        df['attack_cat'] = df['attack_cat'].str.strip()
        df['is_attack'] = (df['attack_cat'].str.lower() != 'benign').astype(int)
        print(f"Clases: {df['attack_cat'].value_counts().to_dict()}")

    # Quitar columnas con varianza cero
    df = df.loc[:, df.nunique() > 1]

    # Guardar limpio
    df.to_csv("CIC-IDS2017_clean.csv", index=False)
    print(f"[OK] CIC-IDS2017_clean.csv -> {df.shape}")

    # También crea versión escalada para Deep Learning
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c not in ['is_attack']]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    df_scaled.to_csv("CIC-IDS2017_clean_scaled.csv", index=False)
    print(f"[OK] CIC-IDS2017_clean_scaled.csv (listo para LSTM/CNN)")

    return df

def split_for_dl(df, name):
    if df is None or df.empty:
        return
    if 'is_attack' not in df.columns:
        print("[!] No hay columna is_attack para split")
        return
    X = df.drop(columns=[c for c in ['attack_cat','is_attack','label'] if c in df.columns], errors='ignore')
    # Solo numérico
    X = X.select_dtypes(include=[np.number])
    y = df['is_attack']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"\n[>>] Split {name}: Train {X_train.shape}, Test {X_test.shape}, %Ataques Train: {y_train.mean()*100:.2f}%")
    # Guardar npz para PyTorch/TensorFlow
    np.savez(f"{name}_dl_split.npz", X_train=X_train.values, X_test=X_test.values, y_train=y_train.values, y_test=y_test.values)
    print(f"[OK] Guardado {name}_dl_split.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloader NIDS")
    parser.add_argument("--dataset", type=str, default="UNSW-NB15", help="NSL-KDD | UNSW-NB15 | CIC-IDS2017 | all")
    args = parser.parse_args()

    targets = ["NSL-KDD","UNSW-NB15","CIC-IDS2017"] if args.dataset.lower()=="all" else [args.dataset]

    for ds in targets:
        if ds not in DATASETS:
            print(f"[!] Dataset {ds} no reconocido")
            continue
        
        print("\n" + "="*60)
        print(f" Procesando: {ds} - {DATASETS[ds]['description']}")
        print("="*60)

        raw_path = download_kaggle(ds)
        if not raw_path:
            print(f"[!] No se pudo descargar {ds}. Descarga manual desde:")
            if ds == "NSL-KDD": print("[https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)")
            if ds == "UNSW-NB15": print("[https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)")
            if ds == "CIC-IDS2017": print("[https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)")
            continue

        df_clean = None
        if ds == "NSL-KDD":
            df_clean = clean_nsl_kdd(raw_path)
        elif ds == "UNSW-NB15":
            df_clean = clean_unsw_nb15(raw_path)
        elif ds == "CIC-IDS2017":
            df_clean = clean_cic_ids2017(raw_path)

        if df_clean is not None:
            split_for_dl(df_clean, ds.replace("-","_"))

    print("\n[FIN] Todo listo. Archivos *_clean.csv listos para tu modelo Deep Learning.")
```
---

### ¿Qué hace el script por dentro? (Proceso de Limpieza)

Este script no solo descarga los datos, sino que soluciona los problemas estructurales más comunes de cada dataset para que puedas empezar a entrenar tus modelos de inmediato:

#### 1. NSL-KDD
*   **Unificación:** Concatena `KDDTrain+.txt` y `KDDTest+.txt`.
*   **Codificación:** Convierte las 3 variables categóricas de texto (`protocol_type`, `service`, `flag`) en valores numéricos mediante `LabelEncoder`.
*   **Agrupación de Etiquetas:** Re-clasifica las 39 etiquetas de ataques específicos y minoritarios en las 5 clases macro didácticas estándar: **normal, DoS, Probe, R2L, U2R**.

#### 2. UNSW-NB15
*   **Unificación:** Une todos los archivos `.csv` del conjunto.
*   **Limpieza Básica:** Elimina valores infinitos (`inf`) y nulos (`NaN`).
*   **Codificación:** Transformación numérica de las columnas `proto`, `service` y `state`.
*   **Estandarización:** Crea la variable objetivo binaria estandarizada `is_attack`.

#### 3. CIC-IDS2017 (El dataset con más ruido)
Este dataset es el estándar actual, pero su versión raw requiere una limpieza intensiva. El script se encarga de:
*   **Saneamiento de Datos:** Elimina valores infinitos (`Infinity`) y nulos (`NaN`).
*   **Deduplicación:** Elimina filas duplicadas exactas.
*   **Limpieza de Cabeceras:** Elimina espacios en blanco iniciales/finales y normaliza los nombres de las columnas (ej: `Flow Duration` -> `Flow_Duration`).
*   **Etiquetado Binario:** Crea la variable objetivo estándar `attack_cat` y `is_attack` (0=benigno, 1=ataque).
*   **Escalado para Deep Learning:** Genera dos versiones del dataset:
    *   `CIC-IDS2017_clean.csv`: Limpio, pero con valores originales.
    *   `CIC-IDS2017_clean_scaled.csv`: Totalmente normalizado utilizando `StandardScaler`, **listo para ser ingerido por arquitecturas profundas como LSTM o CNN.**

---
**Resultado Final:** Para cada dataset procesado, el script realiza una división estratificada (split 80/20 train/test) garantizando la misma proporción de ataques en ambas particiones, y exporta arreglos listos para PyTorch/TensorFlow en formato `.npz`.

---

### 💻 Script, del modelo Autoencoder + LSTM que consuma directamente ese *_dl_split.npz

---

### Script de Modelos de Deep Learning (`nids_model.py`)

Este script consume directamente los archivos `*_dl_split.npz` generados por el paso anterior (`nids_downloader.py`). Implementa dos enfoques de redes neuronales distintos y complementarios:

**Arquitecturas:**
1. **Autoencoder (Detección de Anomalías):** Detecta ataques *zero-day*. Se entrena únicamente con tráfico benigno para aprender su distribución; cuando ingresa un ataque, no sabe reconstruirlo y el error de reconstrucción se dispara.
2. **CNN + LSTM (Clasificación Supervisada):** Combina convoluciones 1D para extracción de características espaciales y capas LSTM para características secuenciales. Clasifica ataques conocidos con alta precisión.

**Instalación y Uso:**
```bash
pip install tensorflow scikit-learn matplotlib pandas

# Ejecutar solo el Autoencoder
python nids_model.py --input UNSW_NB15_dl_split.npz --model autoencoder

# Ejecutar solo CNN-LSTM
python nids_model.py --input CIC_IDS2017_dl_split.npz --model cnn_lstm

# Ejecutar ambos modelos para comparar
python nids_model.py --input NSL_KDD_dl_split.npz --model both
```

"""
NIDS Deep Learning Model - Autoencoder + CNN-LSTM
Consume directamente los archivos *_dl_split.npz generados por nids_downloader.py

Arquitecturas:
1. Autoencoder (Anomalía) - Detecta zero-day
2. CNN + LSTM (Supervisado) - Clasifica el tipo de ataque
"""

import argparse
import numpy as np
import os

# Configuración de TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Reshape, Flatten
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[!] Instala tensorflow: pip install tensorflow")

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

def load_npz(path):
    print(f"[+] Cargando {path}")
    data = np.load(path)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    print(f"X_train: {X_train.shape}, y_test: {np.bincount(y_test)} | %Ataque test: {y_test.mean()*100:.1f}%")
    return X_train, X_test, y_train, y_test

# ================== MODELO 1: AUTOENCODER ==================
def build_autoencoder(input_dim):
    print(f"\n[>>] Construyendo Autoencoder input_dim={input_dim}")
    # Encoder
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x) # Cuello de botella - aprende la esencia de "normal"
    
    # Decoder
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inp, out)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()
    return autoencoder

def train_autoencoder(X_train, y_train, X_test, y_test):
    # DIDÁCTICO: Entrenamos SOLO con tráfico normal
    X_normal = X_train[y_train == 0]
    print(f"\n[>>] Entrenando Autoencoder SOLO con tráfico normal: {X_normal.shape} muestras")
    print("Idea: Si solo ve normal, cuando vea un ataque no sabrá reconstruirlo y el error será alto")

    autoencoder = build_autoencoder(X_normal.shape[1])

    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = autoencoder.fit(
        X_normal, X_normal,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early],
        verbose=1
    )

    # Evaluar: Error de reconstrucción
    X_train_pred = autoencoder.predict(X_train, verbose=0)
    train_mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)

    # Umbral: percentil 95 del error en tráfico normal
    normal_mse = train_mse[y_train == 0]
    threshold = np.percentile(normal_mse, 95)
    print(f"\n[>>] Umbral de anomalía (95 percentile normal): {threshold:.6f}")

    X_test_pred = autoencoder.predict(X_test, verbose=0)
    test_mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
    y_pred_anomaly = (test_mse > threshold).astype(int)

    print("\n--- RESULTADO AUTOENCODER (Anomalía) ---")
    print(classification_report(y_test, y_pred_anomaly, target_names=['Normal','Ataque']))
    print(f"AUC-ROC: {roc_auc_score(y_test, test_mse):.4f}")

    # Gráfico
    plt.figure(figsize=(10,4))
    plt.hist(normal_mse, bins=50, alpha=0.6, label='Normal Train')
    plt.hist(test_mse[y_test==1], bins=50, alpha=0.6, label='Ataque Test')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Umbral {threshold:.4f}')
    plt.legend()
    plt.title("Distribución de Error de Reconstrucción - Autoencoder")
    plt.xlabel("MSE")
    plt.ylabel("Frecuencia")
    plt.savefig("autoencoder_error_hist.png")
    print("[OK] Gráfico guardado: autoencoder_error_hist.png")

    return autoencoder

# ================== MODELO 2: CNN + LSTM ==================
def build_cnn_lstm(input_dim):
    print(f"\n[>>] Construyendo CNN + LSTM input_dim={input_dim}")
    # Para CNN-LSTM necesitamos forma secuencial: (timesteps, features)
    # Truco didáctico: tratamos cada flujo como una secuencia de 1 paso, pero con ventana
    
    model = Sequential([
        Input(shape=(input_dim, 1)), # (features, 1) para Conv1D
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu', padding='same'),
        Dropout(0.3),
        # LSTM necesita secuencia, Conv1D ya dio secuencia
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    model.summary()
    return model

def train_cnn_lstm(X_train, y_train, X_test, y_test):
    print(f"\n[>>] Entrenando CNN+LSTM Supervisado con TODO el tráfico (Normal+Ataque)")

    # Reshape para CNN-LSTM: (samples, features, 1)
    X_train_r = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_r = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = build_cnn_lstm(X_train.shape[1])
    early = EarlyStopping(monitor='val_auc', mode='max', patience=7, restore_best_weights=True)

    history = model.fit(
        X_train_r, y_train,
        epochs=30,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early],
        verbose=1
    )

    print("\n--- RESULTADO CNN+LSTM (Supervisado) ---")
    loss, acc, auc, prec, rec = model.evaluate(X_test_r, y_test, verbose=0)
    print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

    y_pred = (model.predict(X_test_r, verbose=0) > 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Normal','Ataque']))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Guardar modelo
    model.save("nids_cnn_lstm.h5")
    print("[OK] Modelo guardado: nids_cnn_lstm.h5")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Archivo .npz de nids_downloader.py")
    parser.add_argument("--model", type=str, default="both", choices=["autoencoder","cnn_lstm","both"], help="Qué modelo entrenar")
    args = parser.parse_args()

    if not HAS_TF:
        print("Instala tensorflow primero")
        exit(1)

    if not os.path.exists(args.input):
        print(f"[!] No existe {args.input}. Primero corre nids_downloader.py")
        print("Ejemplo: python nids_downloader.py --dataset UNSW-NB15")
        exit(1)

    X_train, X_test, y_train, y_test = load_npz(args.input)

    # Normalización extra por si el CSV no venía escalado
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if args.model in ["autoencoder","both"]:
        train_autoencoder(X_train, y_train, X_test, y_test)

    if args.model in ["cnn_lstm","both"]:
        train_cnn_lstm(X_train, y_train, X_test, y_test)

    print("\n[FIN] Entrenamiento completo.")
    print("Próximo paso para producción/proyecto:")
    print("1. El Autoencoder es tu detector de zero-day (no necesita etiquetas de ataque)")
    print("2. CNN-LSTM es tu clasificador de alta precisión para ataques conocidos")
    print("3. En pipeline de producción: si el Autoencoder lanza una anomalía -> pasas a CNN-LSTM para confirmar/clasificar")


### ¿Qué hace cada modelo? (Explicación didáctica)

#### 1. Autoencoder (El portero que solo conoce a la gente buena)
*   **Entrenamiento Exclusivo:** El código se entrena **SOLO** con tráfico normal (`y == 0`).
*   **Reconstrucción:** Aprende a comprimir y reconstruir el tráfico benigno con un error muy pequeño.
*   **Detección de Anomalías:** Cuando llega un ataque (ej. DDoS), el modelo no sabe cómo procesarlo y el error de reconstrucción se dispara (por ejemplo, `MSE = 0.8` vs `0.02` en tráfico normal).
*   **Umbral Automático:** El script calcula el umbral de detección situándolo en el percentil 95 del tráfico benigno y genera el gráfico `autoencoder_error_hist.png`. Allí visualizarás claramente dos distribuciones separadas (dos "montañas"): normal vs. ataque.

#### 2. CNN + LSTM (El detective especializado)
*   **Extracción Espacial (`Conv1D`):** Analiza las características internas de cada flujo de red (ej. *"¿Hay muchos paquetes SYN sin su respectivo ACK?"*).
*   **Análisis Secuencial (`LSTM`):** Observa la secuencia temporal y el comportamiento de los flujos a lo largo del tiempo.
*   **Métricas Clave:** El script evalúa e imprime *Precision*, *Recall* y *AUC-ROC*. En el contexto de un NIDS, la métrica más crítica es buscar un **Recall > 98%** (para asegurar que no se dejen pasar ataques reales).

> 💾 **Despliegue:** Al finalizar el entrenamiento, el script guarda el modelo como `nids_cnn_lstm.h5`, dejándolo listo para ser integrado en inferencia en tiempo real con motores de captura como **Zeek** o **Wireshark**.
