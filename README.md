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

--
