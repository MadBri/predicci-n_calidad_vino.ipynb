# predicci-n_calidad_vino.ipynb

```bash
wine-project/
├── data/
│   └── winequality-red.csv       # Dataset original (vinos tintos)
│
├── notebooks/
│   ├── 1_exploracion.ipynb      # Análisis exploratorio (EDA)
│   ├── 2_modelo_arbol.ipynb     # Árbol de decisión
│   └── 3_modelo_bosque.ipynb    # Random Forest + optimización
│
└── README.md                    # Este archivo de documentación
```

### Contenido mínimo requerido en cada notebook:

1. **1_exploracion.ipynb**:
   - Carga de datos con Pandas
   - `df.describe()` y gráficos básicos (histogramas/boxplots)
   - Análisis de correlaciones

2. **2_modelo_arbol.ipynb**:
   ```python
   from sklearn.tree import DecisionTreeClassifier
   model = DecisionTreeClassifier(max_depth=5)
   model.fit(X_train, y_train)
   print(classification_report(y_test, model.predict(X_test)))
   ```

3. **3_modelo_bosque.ipynb**:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(n_estimators=100)
   model.fit(X_train, y_train)
   print("Precisión:", model.score(X_test, y_test))
   ```

### README.md mínimo:
```markdown
# Seminario: Predicción de Calidad de Vino

## Resultados
- Random Forest superó al Árbol de Decisión (68% vs 62% precisión)
- Variables determinantes: alcohol, sulfatos y acidez volátil

## Ejecución
1. Instalar dependencias: `pip install pandas scikit-learn matplotlib`
2. Ejecutar notebooks en orden numérico
```

