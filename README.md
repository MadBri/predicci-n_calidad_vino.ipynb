# 🍷 Predicción de Calidad de Vino - Seminario Práctico

## Estructura del Repositorio
```
analisis-calidad-vino/
├── winequality-red.csv          # Archivo de datos inicial
├── notebook_prediccion.ipynb    # Notebook con el análisis detallado
└── README.md                    # Documento introductorio del proyecto
```

## Notebook: `prediccion_vino.ipynb`
Contiene todo el código del trabajo en celdas ejecutables:

```python
# 1. Carga y Análisis Exploratorio
import pandas as pd
df = pd.read_csv('winequality-red.csv', sep=';')
print(df.describe())

# Visualización de distribución de calidad
df['quality'].hist()
```

```python
# 2. Preprocesamiento
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
# 3. Modelo de Árbol de Decisión
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train_scaled, y_train)
print(f"Precisión Árbol: {tree.score(X_test_scaled, y_test):.2f}")
```

```python
# 4. Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
print(f"Precisión Random Forest: {rf.score(X_test_scaled, y_test):.2f}")

# Importancia de características
pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot(kind='barh')
```

## Resultados Clave
| Modelo           | Precisión (Test) |
|------------------|------------------|
| Árbol Decisión   | 0.62             |
| Random Forest    | 0.68             |

## Conclusiones
1. El modelo Random Forest mostró un mejor desempeño en términos de precisión en comparación con el Árbol de Decisión (68% frente a 62%).  
2. Las variables más influyentes en el análisis fueron:
   - Alcohol (12.5%)
   - Sulfatos (9.8%)
   - Acidez volátil (8.3%).  
3. Se identificó un desbalance en el conjunto de datos, con un 75% de las muestras concentradas en las calificaciones de calidad 5 y 6.  

## Requisitos
```bash
pip install pandas scikit-learn matplotlib
```
