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

--
