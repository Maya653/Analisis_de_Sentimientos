# aprendizaje_supervisado.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AprendizajeSupervisadoDBZ:
    def __init__(self):
        # Dataset de reseñas de Dragon Ball Z con etiquetas
        self.reseñas_dbz = [
            # Reseñas positivas (etiqueta = 1)
            ("Dragon Ball Z es increíble, las peleas son épicas y los personajes están geniales", 1),
            ("Me encanta la transformación de Goku en Super Saiyajin, es lo mejor del anime", 1),
            ("Las batallas de DBZ son espectaculares, especialmente contra Cell y Majin Buu", 1),
            ("Vegeta es un personaje fantástico con mucho desarrollo a lo largo de la serie", 1),
            ("La saga de Freezer es obra maestra, tensión y acción en cada episodio", 1),
            ("Gohan siendo el héroe contra Cell fue un momento perfecto e inolvidable", 1),
            ("La música de Dragon Ball Z es excelente y hace las peleas más emocionantes", 1),
            ("Los momentos emotivos como el sacrificio de Vegeta son conmovedores", 1),
            ("Las transformaciones son increíbles y cada una más poderosa que la anterior", 1),
            ("Dragon Ball Z marcó mi infancia, es una serie que nunca olvidaré", 1),
            ("Excelente desarrollo de personajes y narrativa épica", 1),
            ("Los poderes y técnicas especiales son geniales y creativos", 1),
            ("Las peleas tienen una coreografía perfecta y emocionante", 1),
            ("Frieza es uno de los mejores villanos del anime", 1),
            ("El torneo de Cell fue espectacular y lleno de sorpresas", 1),
            
            # Reseñas negativas (etiqueta = 0)
            ("Dragon Ball Z tiene demasiado relleno y episodios que no aportan nada", 0),
            ("Los gritos constantes y el poder creep arruinan la experiencia", 0),
            ("GT fue un desastre total comparado con la calidad de Z", 0),
            ("Los diálogos durante las peleas son repetitivos y aburridos", 0),
            ("Demasiados episodios dedicados solo a cargar ki, es frustrante", 0),
            ("La saga de Buu se alargó innecesariamente y perdió el ritmo", 0),
            ("Los personajes secundarios fueron olvidados después de Z", 0),
            ("Las peleas se volvieron predecibles: gritar más fuerte para ganar", 0),
            ("La animación tiene inconsistencias molestas en varios episodios", 0),
            ("El final de Dragon Ball Z fue decepcionante después de tanto hype", 0),
            ("Muy lento, muchos episodios sin acción real", 0),
            ("Los personajes femeninos están muy mal desarrollados", 0),
            ("Las transformaciones perdieron sentido después de Super Saiyajin", 0),
            ("Demasiado enfoque en el poder y poco en la historia", 0),
            ("Los villanos después de Frieza son aburridos y genéricos", 0)
        ]
        
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=1)
        self.mejor_modelo = None
        self.modelos_entrenados = {}
        self.comentarios_guardados = []
        
    def preprocesar_texto(self, texto):
        """Preprocesa el texto eliminando caracteres especiales y normalizando"""
        texto = texto.lower()
        texto = re.sub(r'[^\w\sáéíóúñü]', '', texto)  # Mantener acentos españoles
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto
    
    def entrenar_modelos(self):
        """Entrena múltiples modelos de aprendizaje supervisado"""
        print("🤖 === ENTRENAMIENTO APRENDIZAJE SUPERVISADO DBZ === 🤖")
        print("=" * 60)
        
        # Preparar datos
        textos = [self.preprocesar_texto(reseña[0]) for reseña in self.reseñas_dbz]
        etiquetas = [reseña[1] for reseña in self.reseñas_dbz]
        
        print(f"📊 Dataset: {len(textos)} reseñas")
        print(f"   - Positivas: {sum(etiquetas)}")
        print(f"   - Negativas: {len(etiquetas) - sum(etiquetas)}")
        
        # Vectorización
        print("\n🔄 Vectorizando texto...")
        X = self.vectorizer.fit_transform(textos)
        print(f"   - Características extraídas: {X.shape[1]}")
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, etiquetas, test_size=0.3, random_state=42, stratify=etiquetas
        )
        
        # Definir modelos a entrenar
        modelos = {
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'SVM Lineal': SVC(kernel='linear', probability=True, random_state=42, C=1.0),
            'Regresión Logística': LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        }
        
        mejor_accuracy = 0
        resultados = {}
        
        print("\n🏋️ Entrenando modelos...")
        print("-" * 50)
        
        for nombre, modelo in modelos.items():
            print(f"\n🔧 Entrenando {nombre}...")
            
            # Entrenar modelo
            modelo.fit(X_train, y_train)
            
            # Hacer predicciones
            y_pred_train = modelo.predict(X_train)
            y_pred_test = modelo.predict(X_test)
            
            # Calcular métricas
            accuracy_train = accuracy_score(y_train, y_pred_train)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            
            # Guardar resultados
            resultados[nombre] = {
                'modelo': modelo,
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'y_test': y_test,
                'y_pred': y_pred_test
            }
            
            print(f"   ✅ Accuracy entrenamiento: {accuracy_train:.3f}")
            print(f"   ✅ Accuracy prueba: {accuracy_test:.3f}")
            
            # Reporte detallado
            print(f"\n   📈 Reporte de clasificación:")
            report = classification_report(y_test, y_pred_test, 
                                         target_names=['😞 Negativo', '😊 Positivo'],
                                         output_dict=True)
            
            print(f"      Precisión Negativo: {report['😞 Negativo']['precision']:.3f}")
            print(f"      Recall Negativo: {report['😞 Negativo']['recall']:.3f}")
            print(f"      Precisión Positivo: {report['😊 Positivo']['precision']:.3f}")
            print(f"      Recall Positivo: {report['😊 Positivo']['recall']:.3f}")
            print(f"      F1-Score Macro: {report['macro avg']['f1-score']:.3f}")
            
            # Seleccionar mejor modelo
            if accuracy_test > mejor_accuracy:
                mejor_accuracy = accuracy_test
                self.mejor_modelo = modelo
                
        self.modelos_entrenados = resultados
        
        print("\n" + "=" * 60)
        print(f"🏆 MEJOR MODELO: {self._obtener_nombre_mejor_modelo()}")
        print(f"🎯 Accuracy: {mejor_accuracy:.3f}")
        
        # Guardar modelo
        self.guardar_modelo()
        
        return resultados
    
    def _obtener_nombre_mejor_modelo(self):
        """Obtiene el nombre del mejor modelo"""
        for nombre, resultado in self.modelos_entrenados.items():
            if resultado['modelo'] == self.mejor_modelo:
                return nombre
        return "No identificado"
    
    def predecir_sentimiento(self, comentario):
        """Predice el sentimiento de un comentario"""
        if self.mejor_modelo is None:
            return {"error": "Modelo no entrenado. Ejecuta entrenar_modelos() primero."}
        
        # Preprocesar comentario
        comentario_limpio = self.preprocesar_texto(comentario)
        
        # Vectorizar
        X_comentario = self.vectorizer.transform([comentario_limpio])
        
        # Predicción
        prediccion = self.mejor_modelo.predict(X_comentario)[0]
        probabilidades = self.mejor_modelo.predict_proba(X_comentario)[0]
        
        # Análisis adicional con TextBlob
        blob = TextBlob(comentario)
        polaridad_textblob = blob.sentiment.polarity
        
        resultado = {
            'comentario_original': comentario,
            'comentario_procesado': comentario_limpio,
            'prediccion': 'Positivo' if prediccion == 1 else 'Negativo',
            'confianza': max(probabilidades),
            'probabilidad_negativo': probabilidades[0],
            'probabilidad_positivo': probabilidades[1],
            'textblob_polaridad': polaridad_textblob,
            'textblob_sentimiento': self._interpretar_textblob(polaridad_textblob),
            'modelo_usado': self._obtener_nombre_mejor_modelo(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return resultado
    
    def _interpretar_textblob(self, polaridad):
        """Interpreta la polaridad de TextBlob"""
        if polaridad > 0.1:
            return "Positivo"
        elif polaridad < -0.1:
            return "Negativo"
        else:
            return "Neutro"
    
    def evaluar_comentario_interactivo(self, comentario):
        """Evalúa un comentario y muestra resultado formateado"""
        resultado = self.predecir_sentimiento(comentario)
        
        if 'error' in resultado:
            print(f"❌ Error: {resultado['error']}")
            return
        
        print(f"\n🐉 ANÁLISIS DE SENTIMIENTO - DRAGON BALL Z 🐉")
        print("=" * 55)
        print(f"💬 Comentario: '{comentario}'")
        print("-" * 55)
        
        # Resultado principal
        emoji = "😊" if resultado['prediccion'] == 'Positivo' else "😞"
        print(f"{emoji} PREDICCIÓN: {resultado['prediccion'].upper()}")
        print(f"🎯 Confianza: {resultado['confianza']:.1%}")
        print(f"🤖 Modelo usado: {resultado['modelo_usado']}")
        
        # Probabilidades detalladas
        print(f"\n📊 Probabilidades:")
        print(f"   😞 Negativo: {resultado['probabilidad_negativo']:.1%}")
        print(f"   😊 Positivo: {resultado['probabilidad_positivo']:.1%}")
        
        # Análisis TextBlob
        tb_emoji = "😊" if resultado['textblob_sentimiento'] == 'Positivo' else "😞" if resultado['textblob_sentimiento'] == 'Negativo' else "😐"
        print(f"\n📝 TextBlob: {tb_emoji} {resultado['textblob_sentimiento']} (Polaridad: {resultado['textblob_polaridad']:.3f})")
        
        # Guardar comentario
        self.comentarios_guardados.append(resultado)
        self.guardar_comentarios()
        
        print(f"💾 Comentario guardado correctamente!")
        print("=" * 55)
        
        return resultado
    
    def probar_multiples_comentarios(self):
        """Permite probar múltiples comentarios interactivamente"""
        comentarios_prueba = [
            "Goku es el mejor protagonista de todos los tiempos",
            "Dragon Ball Z es muy aburrido y repetitivo",
            "Las transformaciones Super Saiyajin son geniales",
            "Demasiados gritos y poco desarrollo de historia",
            "Vegeta tiene el mejor arco de personaje"
        ]
        
        print("\n🧪 PRUEBAS CON COMENTARIOS DE EJEMPLO:")
        print("=" * 50)
        
        for comentario in comentarios_prueba:
            resultado = self.predecir_sentimiento(comentario)
            emoji = "😊" if resultado['prediccion'] == 'Positivo' else "😞"
            print(f"{emoji} '{comentario}' -> {resultado['prediccion']} ({resultado['confianza']:.1%})")
        
        print("\n" + "=" * 50)
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas de los comentarios analizados"""
        if not self.comentarios_guardados:
            print("📊 No hay comentarios analizados aún.")
            return
        
        print(f"\n📊 ESTADÍSTICAS DE ANÁLISIS SUPERVISADO")
        print("=" * 45)
        print(f"Total de comentarios analizados: {len(self.comentarios_guardados)}")
        
        positivos = sum(1 for c in self.comentarios_guardados if c['prediccion'] == 'Positivo')
        negativos = len(self.comentarios_guardados) - positivos
        
        print(f"😊 Comentarios positivos: {positivos} ({positivos/len(self.comentarios_guardados)*100:.1f}%)")
        print(f"😞 Comentarios negativos: {negativos} ({negativos/len(self.comentarios_guardados)*100:.1f}%)")
        
        # Confianza promedio
        confianza_promedio = np.mean([c['confianza'] for c in self.comentarios_guardados])
        print(f"🎯 Confianza promedio: {confianza_promedio:.1%}")
        
        print(f"\n📝 Últimos 5 comentarios:")
        for comentario in self.comentarios_guardados[-5:]:
            emoji = "😊" if comentario['prediccion'] == 'Positivo' else "😞"
            print(f"   {emoji} {comentario['comentario_original'][:40]}... -> {comentario['prediccion']}")
    
    def guardar_modelo(self):
        """Guarda el modelo entrenado"""
        try:
            with open('modelo_supervisado_dbz.pkl', 'wb') as f:
                pickle.dump({
                    'modelo': self.mejor_modelo,
                    'vectorizer': self.vectorizer,
                    'nombre_modelo': self._obtener_nombre_mejor_modelo()
                }, f)
            print("💾 Modelo guardado en 'modelo_supervisado_dbz.pkl'")
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")
    
    def cargar_modelo(self):
        """Carga el modelo previamente entrenado"""
        try:
            with open('modelo_supervisado_dbz.pkl', 'rb') as f:
                datos = pickle.load(f)
                self.mejor_modelo = datos['modelo']
                self.vectorizer = datos['vectorizer']
                print(f"✅ Modelo cargado: {datos['nombre_modelo']}")
                return True
        except FileNotFoundError:
            print("⚠️ No se encontró modelo guardado. Entrena primero.")
            return False
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def guardar_comentarios(self):
        """Guarda los comentarios analizados"""
        try:
            with open('comentarios_supervisado_dbz.json', 'w', encoding='utf-8') as f:
                json.dump(self.comentarios_guardados, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Error guardando comentarios: {e}")

def main():
    """Función principal"""
    print("🐉 SISTEMA DE ANÁLISIS DE SENTIMIENTOS SUPERVISADO - DBZ 🐉")
    print("=" * 65)
    
    analizador = AprendizajeSupervisadoDBZ()
    
    # Intentar cargar modelo existente
    if not analizador.cargar_modelo():
        print("\n🏋️ Entrenando nuevo modelo...")
        analizador.entrenar_modelos()
        analizador.probar_multiples_comentarios()
    
    print("\n🎮 ¡Sistema listo! Comandos disponibles:")
    print("   - Escribe un comentario para analizarlo")
    print("   - 'stats' para ver estadísticas")
    print("   - 'ejemplos' para ver ejemplos de prueba") 
    print("   - 'reentrenar' para entrenar de nuevo")
    print("   - 'salir' para terminar")
    print("=" * 65)
    
    while True:
        comando = input("\n💭 Tu comentario sobre DBZ (o comando): ").strip()
        
        if comando.lower() == 'salir':
            print("¡Hasta la vista! 🐉✨")
            break
        elif comando.lower() == 'stats':
            analizador.mostrar_estadisticas()
        elif comando.lower() == 'ejemplos':
            analizador.probar_multiples_comentarios()
        elif comando.lower() == 'reentrenar':
            print("🔄 Reentrenando modelo...")
            analizador.entrenar_modelos()
            analizador.probar_multiples_comentarios()
        elif comando:
            analizador.evaluar_comentario_interactivo(comando)
        else:
            print("⚠️ Por favor escribe un comentario válido.")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        import textblob
    except ImportError:
        print("📦 Instalando TextBlob...")
        import subprocess
        subprocess.check_call(["pip", "install", "textblob"])
    
    main()