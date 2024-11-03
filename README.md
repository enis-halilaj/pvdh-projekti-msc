# Projekti: Faktorët e performancës së studentëve

Ky projekt është pjesë e lëndës **Përgatitja dhe Vizualizimi i të Dhënave** në kuadër të programit të studimeve Master IKS 2024/25. 
Projekti ynë ka për qëllim të analizojë dhe vizualizojë faktorët që ndikojnë në performancën akademike të studentëve duke përdorur teknika të përpunimit dhe vizualizimit të të dhënave.

Për të realizuar këtë projekt, kemi përdorur këtë dataset: **[Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)**

## Udhëzime në ekzekutimin e projektit
Për të ekzekutuar projektin në pajisjen tuaj, ndiqni këto hapa:

### 1. Kërkesat
Sigurohuni që të keni të instaluar **Python 3.6** ose versionin më të ri. Mund ta shkarkoni nga [python.org](https://www.python.org/downloads/).

### 2. Klonimi i projektit
Klononi projektin me komandën:
```bash
git clone https://github.com/enis-halilaj/pvdh-projekti-msc
```

### 3. Instalimi i paketave
Pasi të jeni në direktoriumin e projektit, instaloni paketat duke përdorur këtë komandë:
```bash
pip3 install -r requirements.txt
```

### 4. Preprocesimi i të dhënave

Në fazën e preprocesimit, të dhënat fillestare janë pastruar dhe përgatitur për analizë. Më poshtë janë disa nga hapat kryesorë qe kemi ndjekur:

#### 4.1 Importimi i librarive

```bash
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
```

#### 4.2 Ngarkimi i të dhënave

```bash
main_df = pd.read_csv("../dataset/StudentPerformanceFactors.csv")
pre_df = pd.read_csv("../dataset/StudentPerformanceFactors_new.csv")
```

#### 4.3 Definimi i tipeve të të dhënave

```bash
print("Tipet e të dhënave: \n")
print(main_df.dtypes)

categorical_columns = main_df.select_dtypes(include=['object']).columns
numerical_columns = main_df.select_dtypes(include=['int64', 'float64']).columns

print("\nAtributet kategorike:", list(categorical_columns))
print("\nAtributet numerike :", list(numerical_columns))
```

#### 4.4 Menaxhimi vlerave të zbrazëta (null):

```bash
null_counts = main_df.isnull().sum()

print(null_counts)
has_nulls = main_df.isnull().any()

print('\nAtributet që kanë vlera të zbrazëta (null):', ', '.join(has_nulls[has_nulls].index))
```

#### 5. Zgjedhja e nen-bashksisë së vetive

Duke përdorur datasetin e ri të gjeneruar, fillojmë me përcaktimin e vetive më të rëndësishme për analizë, duke u fokusuar në ato që janë të lidhura ngushtë me `Exam_Score`.

```bash
features_selected = [
    'Hours_Studied', 
    'Attendance', 
    'Parental_Involvement', 
    'Access_to_Resources', 
    'Extracurricular_Activities', 
    'Sleep_Hours', 
    'Previous_Scores', 
    'Motivation_Level', 
    'Internet_Access', 
    'Tutoring_Sessions', 
    'Family_Income', 
    'Teacher_Quality', 
    'Peer_Influence', 
    'Physical_Activity', 
    'Gender'
]

df_selected_features = pre_df[features_selected + ['Exam_Score']]

print("Dataframe me vetitë e zgjedhura:")
display(df_selected_features)
```

#### 6. Krijimi i vetive të reja

Një prej vetive të cilat do të krijojmë është `Study_Value`, e cila është një prodhim i `Hours_Studied` dhe `Previous_Scores`. Ky atribut do të ndihmojë për të kuptuar se sa efektive janë orët e studimit në lidhje me rezultatet e mëparshme.

```bash
df_selected_features.loc[:, 'Study_Value'] = df_selected_features['Hours_Studied'] * df_selected_features['Previous_Scores']

print("Dataframe pas krijimit të Study_Value:")
display(df_selected_features[['Hours_Studied', 'Previous_Scores', 'Study_Value']].head())
```

Një tjetër veti e re do të jetë `Activity_Score`, e cila është një shprehje për të kombinuar disa aktivitete fizike dhe jashteshkollore. Përdoreshim `Physical_Activity` dhe `Extracurricular_Activities` për të krijuar këtë veti.

```bash
df_selected_features.loc[:, 'Activity_Score'] = df_selected_features['Physical_Activity'] + df_selected_features['Extracurricular_Activities'].apply(lambda x: 1 if x == 'Yes' else 0)

print("Dataframe pas krijimit të Activity_Score:")
display(df_selected_features[['Physical_Activity', 'Extracurricular_Activities', 'Activity_Score']].head())
```