# aprendizaje_no_supervisado.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
import json
import pickle
from datetime import datetime
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class AprendizajeNoSupervisadoDBZ:
    def __init__(self):
        # Dataset de reseñas de Dragon Ball Z (SIN etiquetas para simular no supervisado)
        self.reseñas_dbz = [
            # Reseñas positivas (15)
            "Dragon Ball Z es increíble, las peleas son épicas y los personajes están geniales",
            "Me encanta la transformación de Goku en Super Saiyajin, es lo mejor del anime",
            "Las batallas de DBZ son espectaculares, especialmente contra Cell y Majin Buu",
            "Vegeta es un personaje fantástico con mucho desarrollo a lo largo de la serie",
            "La saga de Freezer es obra maestra, tensión y acción en cada episodio",
            "Gohan siendo el héroe contra Cell fue un momento perfecto e inolvidable",
            "La música de Dragon Ball Z es excelente y hace las peleas más emocionantes",
            "Los momentos emotivos como el sacrificio de Vegeta son conmovedores",
            "Las transformaciones son increíbles y cada una más poderosa que la anterior",
            "Dragon Ball Z marcó mi infancia, es una serie que nunca olvidaré",
            "Excelente desarrollo de personajes y narrativa épica",
            "Los poderes y técnicas especiales son geniales y creativos",
            "Las peleas tienen una coreografía perfecta y emocionante",
            "Frieza es uno de los mejores villanos del anime",
            "El torneo de Cell fue espectacular y lleno de sorpresas",
            
            # Reseñas negativas (20)
            "Dragon Ball Z tiene demasiado relleno y episodios que no aportan nada",
            "Los gritos constantes y el poder creep arruinan la experiencia",
            "GT fue un desastre total comparado con la calidad de Z",
            "Los diálogos durante las peleas son repetitivos y aburridos",
            "Demasiados episodios dedicados solo a cargar ki, es frustrante",
            "La saga de Buu se alargó innecesariamente y perdió el ritmo",
            "Los personajes secundarios fueron olvidados después de Z",
            "Las peleas se volvieron predecibles: gritar más fuerte para ganar",
            "La animación tiene inconsistencias molestas en varios episodios",
            "El final de Dragon Ball Z fue decepcionante después de tanto hype",
            "Muy lento, muchos episodios sin acción real",
            "Los personajes femeninos están muy mal desarrollados",
            "Las transformaciones perdieron sentido después de Super Saiyajin",
            "Demasiado enfoque en el poder y poco en la historia",
            "Los villanos después de Frieza son aburridos y genéricos",
            "La serie se volvió muy comercial y perdió su esencia",
            "Muchos plot holes y inconsistencias en la historia",
            "Los combates son demasiado largos y tediosos",
            "Falta de desarrollo para personajes no Saiyajin",
            "El poder scaling no tiene sentido después de Namek"
        ]
        
        # Etiquetas reales (solo para evaluación, NO se usan en entrenamiento)
        self.etiquetas_reales = [1]*15 + [0]*20  # 15 positivas, 20 negativas = 35 total = 35 total
        
        self.vectorizer = TfidfVectorizer(max_features=800, ngram_range=(1, 2), min_df=1)
        self.modelo_clustering = None
        self.clusters = None
        self.X_vectorizado = None
        self.comentarios_analizados = []
        self.interpretacion_clusters = {}
        
    def preprocesar_texto(self, texto):
        """Preprocesa el texto eliminando caracteres especiales y normalizando"""
        texto = texto.lower()
        texto = re.sub(r'[^\w\sáéíóúñü]', '', texto)  # Mantener acentos españoles
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto
    
    def entrenar_clustering(self):
        """Entrena diferentes modelos de clustering"""
        print("🔍 === ENTRENAMIENTO APRENDIZAJE NO SUPERVISADO DBZ === 🔍")
        print("=" * 65)
        
        # Preprocesar textos
        textos_procesados = [self.preprocesar_texto(texto) for texto in self.reseñas_dbz]
        
        print(f"📊 Dataset: {len(textos_procesados)} reseñas (SIN etiquetas)")
        
        # Vectorización
        print("\n🔄 Vectorizando texto...")
        self.X_vectorizado = self.vectorizer.fit_transform(textos_procesados)
        print(f"   - Características extraídas: {self.X_vectorizado.shape[1]}")
        
        # Probar diferentes algoritmos de clustering
        algoritmos = {
            'K-Means': KMeans(n_clusters=2, random_state=42, n_init=10),
            'K-Means (3 clusters)': KMeans(n_clusters=3, random_state=42, n_init=10),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=2),
            'Clustering Jerárquico': AgglomerativeClustering(n_clusters=2)
        }
        
        resultados = {}
        
        print("\n🔬 Probando algoritmos de clustering...")
        print("-" * 50)
        
        for nombre, algoritmo in algoritmos.items():
            print(f"\n🧪 Aplicando {nombre}...")
            
            # Aplicar clustering
            if nombre == 'DBSCAN' or 'Jerárquico' in nombre:
                # DBSCAN y Clustering Jerárquico necesitan datos densos
                X_denso = self.X_vectorizado.toarray()
                clusters = algoritmo.fit_predict(X_denso)
            else:
                # K-Means puede trabajar con matrices dispersas
                clusters = algoritmo.fit_predict(self.X_vectorizado)
            
            # Calcular métricas
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_ruido = list(clusters).count(-1)
            
            # Silhouette score (solo si hay más de 1 cluster)
            if n_clusters > 1 and n_clusters < len(textos_procesados):
                if nombre == 'DBSCAN' or 'Jerárquico' in nombre:
                    # Ya tenemos X_denso para estos algoritmos
                    silhouette = silhouette_score(X_denso, clusters) if n_clusters > 1 else -1
                else:
                    # Para K-Means convertimos a denso
                    silhouette = silhouette_score(self.X_vectorizado.toarray(), clusters)
            else:
                silhouette = -1
            
            resultados[nombre] = {
                'algoritmo': algoritmo,
                'clusters': clusters,
                'n_clusters': n_clusters,
                'n_ruido': n_ruido,
                'silhouette_score': silhouette
            }
            
            print(f"   ✅ Clusters encontrados: {n_clusters}")
            if n_ruido > 0:
                print(f"   ⚠️ Puntos de ruido: {n_ruido}")
            if silhouette > -1:
                print(f"   📈 Silhouette Score: {silhouette:.3f}")
            
            # Mostrar distribución de clusters
            cluster_counts = Counter(clusters)
            print(f"   📊 Distribución: {dict(cluster_counts)}")
        
        # Seleccionar mejor algoritmo (K-Means con 2 clusters por defecto)
        self.modelo_clustering = algoritmos['K-Means']
        self.clusters = resultados['K-Means']['clusters']
        
        print(f"\n🏆 Algoritmo seleccionado: K-Means (2 clusters)")
        print(f"🎯 Silhouette Score: {resultados['K-Means']['silhouette_score']:.3f}")
        
        # Interpretar clusters
        self._interpretar_clusters()
        
        # Visualizar resultados
        self._visualizar_clusters()
        
        # Guardar modelo
        self._guardar_modelo()
        
        return resultados
    
    def _interpretar_clusters(self):
        """Interpreta el significado de cada cluster usando análisis de palabras"""
        print(f"\n🧠 INTERPRETACIÓN DE CLUSTERS")
        print("=" * 40)
        
        textos_procesados = [self.preprocesar_texto(texto) for texto in self.reseñas_dbz]
        
        for cluster_id in set(self.clusters):
            print(f"\n🔍 CLUSTER {cluster_id}:")
            
            # Obtener textos del cluster
            indices_cluster = [i for i, c in enumerate(self.clusters) if c == cluster_id]
            textos_cluster = [textos_procesados[i] for i in indices_cluster]
            
            print(f"   📊 Total de reseñas: {len(textos_cluster)}")
            
            # Analizar palabras más frecuentes
            texto_completo = ' '.join(textos_cluster)
            palabras = texto_completo.split()
            palabras_frecuentes = Counter(palabras).most_common(10)
            
            print(f"   🔤 Palabras más frecuentes:")
            for palabra, freq in palabras_frecuentes[:5]:
                print(f"      - {palabra}: {freq}")
            
            # Análisis de sentimiento promedio con TextBlob
            polaridades = []
            for texto_original in [self.reseñas_dbz[i] for i in indices_cluster]:
                blob = TextBlob(texto_original)
                polaridades.append(blob.sentiment.polarity)
            
            polaridad_promedio = np.mean(polaridades)
            
            # Interpretación del cluster
            if polaridad_promedio > 0.05:
                interpretacion = "😊 Positivo"
                sentimiento = "positivo"
            elif polaridad_promedio < -0.05:
                interpretacion = "😞 Negativo"
                sentimiento = "negativo"
            else:
                interpretacion = "😐 Neutro"
                sentimiento = "neutro"
            
            self.interpretacion_clusters[cluster_id] = sentimiento
            
            print(f"   🎭 Polaridad promedio: {polaridad_promedio:.3f}")
            print(f"   🏷️ Interpretación: {interpretacion}")
            
            # Mostrar ejemplos representativos
            print(f"   📝 Ejemplos de reseñas:")
            for i, idx in enumerate(indices_cluster[:3]):
                print(f"      {i+1}. {self.reseñas_dbz[idx][:60]}...")
    
    def _visualizar_clusters(self):
        """Visualiza los clusters usando PCA y gráficos"""
        print(f"\n📊 VISUALIZACIÓN DE CLUSTERS")
        print("=" * 35)
        
        # PCA para reducir dimensionalidad
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(self.X_vectorizado.toarray())
        
        # Crear visualización
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Clusters predichos
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.clusters, 
                            cmap='viridis', alpha=0.7, s=100)
        plt.title('🔍 Clusters Descubiertos\n(Aprendizaje No Supervisado)', fontsize=12)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
        plt.colorbar(scatter, label='Cluster ID')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Sentimientos reales (para comparación)
        plt.subplot(1, 3, 2)
        colores_reales = ['red' if label == 0 else 'green' for label in self.etiquetas_reales]
        scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colores_reales, 
                             alpha=0.7, s=100)
        plt.title('🎯 Sentimientos Reales\n(Solo para comparación)', fontsize=12)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
        plt.grid(True, alpha=0.3)
        
        # Añadir leyenda manual
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='😞 Negativo'),
                          Patch(facecolor='green', label='😊 Positivo')]
        plt.legend(handles=legend_elements)
        
        # Subplot 3: Comparación de precisión
        plt.subplot(1, 3, 3)
        self._calcular_precision_clusters()
        
        plt.tight_layout()
        plt.savefig('clusters_dbz_no_supervisado.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Gráfico guardado como 'clusters_dbz_no_supervisado.png'")
    
    def _calcular_precision_clusters(self):
        """Calcula qué tan bien los clusters capturan los sentimientos"""
        # Crear matriz de confusión entre clusters y sentimientos reales
        cluster_0_real = [self.etiquetas_reales[i] for i in range(len(self.clusters)) if self.clusters[i] == 0]
        cluster_1_real = [self.etiquetas_reales[i] for i in range(len(self.clusters)) if self.clusters[i] == 1]
        
        # Calcular pureza de cada cluster
        pureza_cluster_0 = max(cluster_0_real.count(0), cluster_0_real.count(1)) / len(cluster_0_real) if cluster_0_real else 0
        pureza_cluster_1 = max(cluster_1_real.count(0), cluster_1_real.count(1)) / len(cluster_1_real) if cluster_1_real else 0
        
        clusters = ['Cluster 0', 'Cluster 1']
        purezas = [pureza_cluster_0, pureza_cluster_1]
        colores = ['lightcoral', 'lightgreen']
        
        bars = plt.bar(clusters, purezas, color=colores, alpha=0.7, edgecolor='black')
        plt.title('🎯 Pureza de Clusters\n(% de elementos dominantes)', fontsize=12)
        plt.ylabel('Pureza (%)')
        plt.ylim(0, 1)
        
        # Añadir valores en las barras
        for bar, pureza in zip(bars, purezas):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{pureza:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
    
    def predecir_cluster(self, comentario):
        """Predice el cluster de un nuevo comentario"""
        if self.modelo_clustering is None:
            return {"error": "Modelo no entrenado. Ejecuta entrenar_clustering() primero."}
        
        # Preprocesar comentario
        comentario_limpio = self.preprocesar_texto(comentario)
        
        # Vectorizar
        X_comentario = self.vectorizer.transform([comentario_limpio])
        
        # Predicir cluster
        cluster_predicho = self.modelo_clustering.predict(X_comentario)[0]
        
        # Calcular distancia al centroide
        centroide = self.modelo_clustering.cluster_centers_[cluster_predicho]
        distancia = np.linalg.norm(X_comentario.toarray() - centroide)
        
        # Análisis adicional con TextBlob
        blob = TextBlob(comentario)
        polaridad_textblob = blob.sentiment.polarity
        
        # Interpretar resultado
        sentimiento_interpretado = self.interpretacion_clusters.get(cluster_predicho, "desconocido")
        
        resultado = {
            'comentario_original': comentario,
            'comentario_procesado': comentario_limpio,
            'cluster_predicho': int(cluster_predicho),
            'sentimiento_interpretado': sentimiento_interpretado,
            'distancia_centroide': float(distancia),
            'confianza': max(0, 1 - (distancia / 2)),  # Confianza basada en distancia
            'textblob_polaridad': polaridad_textblob,
            'textblob_sentimiento': self._interpretar_textblob(polaridad_textblob),
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
        resultado = self.predecir_cluster(comentario)
        
        if 'error' in resultado:
            print(f"❌ Error: {resultado['error']}")
            return
        
        print(f"\n🔍 ANÁLISIS NO SUPERVISADO - DRAGON BALL Z 🔍")
        print("=" * 55)
        print(f"💬 Comentario: '{comentario}'")
        print("-" * 55)
        
        # Resultado principal
        cluster_emoji = "😊" if resultado['sentimiento_interpretado'] == 'positivo' else "😞" if resultado['sentimiento_interpretado'] == 'negativo' else "😐"
        print(f"{cluster_emoji} CLUSTER ASIGNADO: {resultado['cluster_predicho']}")
        print(f"🎭 SENTIMIENTO INTERPRETADO: {resultado['sentimiento_interpretado'].upper()}")
        print(f"🎯 Confianza: {resultado['confianza']:.1%}")
        print(f"📏 Distancia al centroide: {resultado['distancia_centroide']:.3f}")
        
        # Análisis TextBlob
        tb_emoji = "😊" if resultado['textblob_sentimiento'] == 'Positivo' else "😞" if resultado['textblob_sentimiento'] == 'Negativo' else "😐"
        print(f"\n📝 TextBlob: {tb_emoji} {resultado['textblob_sentimiento']} (Polaridad: {resultado['textblob_polaridad']:.3f})")
        
        # Consistencia entre métodos
        consistente = (
            (resultado['sentimiento_interpretado'] == 'positivo' and resultado['textblob_sentimiento'] == 'Positivo') or
            (resultado['sentimiento_interpretado'] == 'negativo' and resultado['textblob_sentimiento'] == 'Negativo')
        )
        
        consistency_emoji = "✅" if consistente else "⚠️"
        print(f"{consistency_emoji} Consistencia entre métodos: {'SÍ' if consistente else 'NO'}")
        
        # Guardar comentario
        self.comentarios_analizados.append(resultado)
        self._guardar_comentarios()
        
        print(f"💾 Comentario guardado correctamente!")
        print("=" * 55)
        
        return resultado
    
    def generar_nube_palabras(self):
        """Genera nubes de palabras para cada cluster"""
        try:
            print("\n☁️ GENERANDO NUBES DE PALABRAS POR CLUSTER...")
            
            textos_procesados = [self.preprocesar_texto(texto) for texto in self.reseñas_dbz]
            
            plt.figure(figsize=(15, 6))
            
            for cluster_id in set(self.clusters):
                indices_cluster = [i for i, c in enumerate(self.clusters) if c == cluster_id]
                textos_cluster = [textos_procesados[i] for i in indices_cluster]
                texto_completo = ' '.join(textos_cluster)
                
                # Crear nube de palabras
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white',
                                    colormap='viridis' if cluster_id == 0 else 'plasma',
                                    max_words=50).generate(texto_completo)
                
                plt.subplot(1, 2, cluster_id + 1)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Cluster {cluster_id}\n({self.interpretacion_clusters.get(cluster_id, "desconocido")})')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('wordclouds_clusters_dbz.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("☁️ Nubes de palabras guardadas como 'wordclouds_clusters_dbz.png'")
            
        except ImportError:
            print("⚠️ WordCloud no está instalado. Instala con: pip install wordcloud")
        except Exception as e:
            print(f"❌ Error generando nubes de palabras: {e}")
    
    def analizar_topicos(self, n_topicos=3):
        """Realiza análisis de tópicos usando LDA"""
        print(f"\n📚 ANÁLISIS DE TÓPICOS (LDA)")
        print("=" * 35)
        
        # Usar CountVectorizer para LDA
        vectorizer_lda = CountVectorizer(max_features=100, min_df=1, max_df=0.8)
        textos_procesados = [self.preprocesar_texto(texto) for texto in self.reseñas_dbz]
        X_counts = vectorizer_lda.fit_transform(textos_procesados)
        
        # Aplicar LDA
        lda = LatentDirichletAllocation(n_components=n_topicos, random_state=42, max_iter=100)
        lda.fit(X_counts)
        
        # Mostrar tópicos
        feature_names = vectorizer_lda.get_feature_names_out()
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"\n🎯 Tópico {topic_idx + 1}:")
            print(f"   Palabras clave: {', '.join(top_words[:5])}")
            
        return lda, vectorizer_lda
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas de los comentarios analizados"""
        if not self.comentarios_analizados:
            print("📊 No hay comentarios analizados aún.")
            return
        
        print(f"\n📊 ESTADÍSTICAS DE ANÁLISIS NO SUPERVISADO")
        print("=" * 50)
        print(f"Total de comentarios analizados: {len(self.comentarios_analizados)}")
        
        # Distribución por clusters
        clusters_asignados = [c['cluster_predicho'] for c in self.comentarios_analizados]
        cluster_counts = Counter(clusters_asignados)
        
        for cluster_id, count in cluster_counts.items():
            sentimiento = self.interpretacion_clusters.get(cluster_id, "desconocido")
            emoji = "😊" if sentimiento == 'positivo' else "😞" if sentimiento == 'negativo' else "😐"
            porcentaje = count / len(self.comentarios_analizados) * 100
            print(f"{emoji} Cluster {cluster_id} ({sentimiento}): {count} ({porcentaje:.1f}%)")
        
        # Confianza promedio
        confianza_promedio = np.mean([c['confianza'] for c in self.comentarios_analizados])
        print(f"🎯 Confianza promedio: {confianza_promedio:.1%}")
        
        # Distancia promedio
        distancia_promedio = np.mean([c['distancia_centroide'] for c in self.comentarios_analizados])
        print(f"📏 Distancia promedio al centroide: {distancia_promedio:.3f}")
        
        print(f"\n📝 Últimos 5 comentarios:")
        for comentario in self.comentarios_analizados[-5:]:
            cluster_id = comentario['cluster_predicho']
            sentimiento = self.interpretacion_clusters.get(cluster_id, "desconocido")
            emoji = "😊" if sentimiento == 'positivo' else "😞" if sentimiento == 'negativo' else "😐"
            print(f"   {emoji} {comentario['comentario_original'][:40]}... -> Cluster {cluster_id}")
    
    def _guardar_modelo(self):
        """Guarda el modelo entrenado"""
        try:
            with open('modelo_no_supervisado_dbz.pkl', 'wb') as f:
                pickle.dump({
                    'modelo_clustering': self.modelo_clustering,
                    'vectorizer': self.vectorizer,
                    'interpretacion_clusters': self.interpretacion_clusters,
                    'clusters_entrenamiento': self.clusters
                }, f)
            print("💾 Modelo no supervisado guardado en 'modelo_no_supervisado_dbz.pkl'")
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")
    
    def _cargar_modelo(self):
        """Carga el modelo previamente entrenado"""
        try:
            with open('modelo_no_supervisado_dbz.pkl', 'rb') as f:
                datos = pickle.load(f)
                self.modelo_clustering = datos['modelo_clustering']
                self.vectorizer = datos['vectorizer']
                self.interpretacion_clusters = datos['interpretacion_clusters']
                self.clusters = datos['clusters_entrenamiento']
                print("✅ Modelo no supervisado cargado correctamente")
                return True
        except FileNotFoundError:
            print("⚠️ No se encontró modelo guardado. Entrena primero.")
            return False
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def _guardar_comentarios(self):
        """Guarda los comentarios analizados"""
        try:
            with open('comentarios_no_supervisado_dbz.json', 'w', encoding='utf-8') as f:
                json.dump(self.comentarios_analizados, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Error guardando comentarios: {e}")
    
    def probar_ejemplos(self):
        """Prueba el modelo con comentarios de ejemplo"""
        ejemplos = [
            "Goku es el mejor héroe de todos los tiempos, increíble",
            "Dragon Ball Z es muy aburrido y lento",
            "Las transformaciones Super Saiyajin son espectaculares",
            "Demasiados episodios de relleno sin sentido",
            "Vegeta tiene el mejor desarrollo de personaje"
        ]
        
        print("\n🧪 PRUEBAS CON EJEMPLOS:")
        print("=" * 40)
        
        for comentario in ejemplos:
            resultado = self.predecir_cluster(comentario)
            cluster_id = resultado['cluster_predicho']
            sentimiento = self.interpretacion_clusters.get(cluster_id, "desconocido")
            emoji = "😊" if sentimiento == 'positivo' else "😞" if sentimiento == 'negativo' else "😐"
            print(f"{emoji} '{comentario[:40]}...' -> Cluster {cluster_id} ({sentimiento})")

def main():
    """Función principal"""
    print("🔍 SISTEMA DE ANÁLISIS NO SUPERVISADO - DRAGON BALL Z 🔍")
    print("=" * 65)
    
    analizador = AprendizajeNoSupervisadoDBZ()
    
    # Intentar cargar modelo existente
    if not analizador._cargar_modelo():
        print("\n🔬 Entrenando modelo de clustering...")
        analizador.entrenar_clustering()
        
        # Análisis adicionales
        analizador.generar_nube_palabras()
        analizador.analizar_topicos()
        analizador.probar_ejemplos()
    
    print("\n🎮 ¡Sistema listo! Comandos disponibles:")
    print("   - Escribe un comentario para analizarlo")
    print("   - 'stats' para ver estadísticas")
    print("   - 'ejemplos' para ver ejemplos de prueba")
    print("   - 'topicos' para análisis de tópicos")
    print("   - 'nubes' para generar nubes de palabras")
    print("   - 'reentrenar' para entrenar de nuevo")
    print("   - 'salir' para terminar")
    print("=" * 65)
    
    while True:
        comando = input("\n💭 Tu comentario sobre DBZ (o comando): ").strip()
        
        if comando.lower() == 'salir':
            print("¡Hasta la vista! 🔍✨")
            break
        elif comando.lower() == 'stats':
            analizador.mostrar_estadisticas()
        elif comando.lower() == 'ejemplos':
            analizador.probar_ejemplos()
        elif comando.lower() == 'topicos':
            analizador.analizar_topicos()
        elif comando.lower() == 'nubes':
            analizador.generar_nube_palabras()
        elif comando.lower() == 'reentrenar':
            print("🔄 Reentrenando modelo...")
            analizador.entrenar_clustering()
            analizador.generar_nube_palabras()
            analizador.analizar_topicos()
        elif comando:
            analizador.evaluar_comentario_interactivo(comando)
        else:
            print("⚠️ Por favor escribe un comentario válido.")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        import wordcloud
    except ImportError:
        print("📦 WordCloud no encontrado. Instala con: pip install wordcloud")
    
    try:
        import textblob
    except ImportError:
        print("📦 Instalando TextBlob...")
        import subprocess
        subprocess.check_call(["pip", "install", "textblob"])
    
    main()