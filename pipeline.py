import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Caricamento del dataset
percorso_file = "dataset.csv" 
print("Caricamento dati in corso... (potrebbe volerci un po' per file grandi!)\n")
df = pd.read_csv(percorso_file)

# Pulizia dei nomi delle colonne (facciamolo subito per non impazzire con gli spazi)
df.columns = df.columns.str.strip()

# --- NUOVA FASE: ESPLORAZIONE E VERIFICA DEI DATI ---
print("=== FASE 1: ESPLORAZIONE DEL DATASET ===")
print(f"Dimensioni iniziali (Righe, Colonne): {df.shape}\n")

print("1. Prime 5 righe del dataset (sguardo rapido):")
print(df.head(), "\n")

print("2. Distribuzione iniziale delle classi (Target):")
print(df['Label'].value_counts())
print("======================================\n")

# --- FINE ESPLORAZIONE, INIZIO PREPROCESSING ---
print("\n=== FASE 2/1: PREPROCESSING ===")
# 1. Gestione dei valori anomali (Pulizia vera e propria)
print("Inizio la pulizia dei dati (rimozione NaN e Inf)...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Dimensioni del dataset DOPO la pulizia: {df.shape}\n")

# 2. Separazione Feature (X) e Target (y)
X = df.drop(columns=['Label'])
y = df['Label']

# 3. Train/Test Split Stratificato
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

print(f"Dimensioni Training set (X): {X_train.shape}")
print(f"Dimensioni Test set (X): {X_test.shape}")
print("======================================\n")



from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt

print("\n=== FASE 2/2: FEATURE SELECTION (Filter Method) ===")
print("Calcolo dell'Information Gain (Mutual Information) in corso...")

# Scegliamo di tenere le 20 migliori feature
k_best = 20
selector = SelectKBest(score_func=mutual_info_classif, k=k_best)

# FIT e TRANSFORM sul Training Set (il modello impara e applica la riduzione)
X_train_selected = selector.fit_transform(X_train, y_train)

# SOLO TRANSFORM sul Test Set (applichiamo la stessa riduzione senza "barare")
X_test_selected = selector.transform(X_test)

# Recuperiamo i nomi delle feature vincenti
colonne_selezionate = X_train.columns[selector.get_support()]

print(f"\nSelezione completata! Da 78 colonne siamo passati a {k_best}.")
print("\nEcco le Top Feature che useremo per addestrare l'albero:")
for i, col in enumerate(colonne_selezionate, 1):
    print(f"{i}. {col}")

# Opzionale: Stampiamo anche i punteggi per vedere quanto distaccano le prime dalle ultime
scores = selector.scores_[selector.get_support()]
feature_scores = pd.DataFrame({'Feature': colonne_selezionate, 'Score': scores})
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

print("\nprimi 5 punteggi:")
print(feature_scores.head())
print("======================================\n")


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

print("\n=== FASE 3: ADDESTRAMENTO E OTTIMIZZAZIONE (Grid Search) ===")
# Impostiamo la griglia dei parametri da testare
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 10, 20] # Quanti campioni servono minimo per creare una nuova ramificazione
}

# Creiamo il modello base (mettiamo una random_state per la riproducibilità)
dt = DecisionTreeClassifier(random_state=42)

# Configuriamo la Grid Search con Cross Validation a 5 fold
print("Ricerca dei parametri migliori in corso (GridSearchCV)...")
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted', # Ottimizziamo per la metrica che ci interessa davvero
    n_jobs=-1
)

# Addestriamo la Grid Search sulle 20 top feature
grid_search.fit(X_train_selected, y_train)

print("\n🏆 Addestramento completato!")
print(f"I parametri vincenti scelti dall'algoritmo sono: {grid_search.best_params_}")
print(f"Miglior F1-score medio durante la Cross-Validation: {grid_search.best_score_:.4f}")

# Salviamo il modello "campione"
best_tree = grid_search.best_estimator_

print("\n=== FASE 4: VALUTAZIONE SUL TEST SET ===")
# Facciamo fare le predizioni al modello migliore sui dati di test (che non ha mai visto)
y_pred = best_tree.predict(X_test_selected)

# Stampiamo il report delle metriche
print("\nReport di Classificazione Finale:")
print(classification_report(y_test, y_pred, digits=4))

# Disegniamo la Matrice di Confusione
print("Generazione della Matrice di Confusione in corso...")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title("Matrice di Confusione - Decision Tree (Top 20 Features)")
plt.tight_layout()
plt.show()