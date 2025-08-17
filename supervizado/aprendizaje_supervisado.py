# supervisado.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# === Cargar dataset desde el CSV ===
df = pd.read_csv("reseñas_dbz.csv")

# Entrenar modelo supervisado
modelo = make_pipeline(CountVectorizer(), MultinomialNB())
modelo.fit(df["reseña"], df["sentimiento"])

# Guardar modelo entrenado
joblib.dump(modelo, "modelo_supervisado.pkl")

print("✅ Modelo supervisado entrenado con reseñas de Dragon Ball Z.")

# === Interacción con el usuario ===
while True:
    comentario = input("\nEscribe tu reseña de Dragon Ball Z (o 'salir' para terminar): ")
    if comentario.lower() == "salir":
        print("👋 Saliendo del análisis de sentimientos.")
        break

    prediccion = modelo.predict([comentario])[0]
    print(f"📊 Análisis de sentimientos: {prediccion.upper()}")

    # Guardar en archivo
    with open("reseñas_guardadas.csv", "a", encoding="utf-8") as f:
        f.write(f"{comentario},{prediccion}\n")
