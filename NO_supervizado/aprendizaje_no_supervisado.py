# no_supervisado.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

# === Cargar dataset desde el CSV ===
df = pd.read_csv("reseñas_dbz.csv")
reseñas = df["reseña"].tolist()

# Convertir a vectores TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reseñas)

# Aplicar KMeans (2 clusters: positivo/negativo)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

# Guardar modelo
joblib.dump((kmeans, vectorizer), "modelo_no_supervisado.pkl")

print("✅ Modelo no supervisado entrenado con reseñas de Dragon Ball Z.")

# === Interacción con el usuario ===
while True:
    comentario = input("\nEscribe tu reseña de Dragon Ball Z (o 'salir' para terminar): ")
    if comentario.lower() == "salir":
        print("👋 Saliendo del análisis de sentimientos.")
        break

    # Transformar comentario
    X_new = vectorizer.transform([comentario])
    cluster = kmeans.predict(X_new)[0]

    # Para decidir si el cluster es positivo o negativo
    # Buscamos cuál cluster tiene más positivos en el dataset inicial
    df["cluster"] = kmeans.predict(X)
    cluster_mayoria = df.groupby("cluster")["sentimiento"].apply(lambda x: x.value_counts().idxmax())

    sentimiento = cluster_mayoria[cluster]
    print(f"📊 Análisis de sentimientos (no supervisado): {sentimiento.upper()}")

    # Guardar en archivo
    with open("reseñas_guardadas.csv", "a", encoding="utf-8") as f:
        f.write(f"{comentario},{sentimiento}\n")
