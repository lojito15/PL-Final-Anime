from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import requests
import json

from pyspark.sql.types import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# crear sesion spark
spark = SparkSession.builder.appName("Practica Final").getOrCreate()

# CARGA Y LIMPIEZA DE DATOS
# cargar los datos con indicaciones para funcionar correctamente
# header --> nombres columnas, inferSchema --> detectar tipo de columna, multiline --> manejo de titulos con comas, encoding --> codificacion caracteres, escape --> manejo comillas
animeCSV = spark.read.csv("anime.csv", header=True, inferSchema=True, multiLine=True, encoding="UTF-8",escape='"')

ratingCSV = spark.read.csv("rating_complete.csv", header=True, inferSchema=True, encoding="UTF-8",)

valoracionesCSV= spark.read.csv('valoraciones_EP.csv',header=False, inferSchema=True, encoding="UTF-8").toDF("user_id", "anime_id", "rating")

# Visualizar primeras filas de csv anime
animeCSV.printSchema()
animeCSV.show(10)

# Visualizar primeras filas de csv rating
ratingCSV.printSchema()
ratingCSV.show(10)

# Visualizar primeras filas de csv valoraciones EP
valoracionesCSV.printSchema()
valoracionesCSV.show(10)

# lista para almacenar las columnas que se quieren eliminar
columnas_eliminar = [
    "Japanese name",
    "Aired",
    "Premiered",
    "Episodes",
    "Duration",
    "Rating",
    "Source",
    "Popularity",
    "Favorites",
    "Watching",
    "On-Hold",
    "Plan to Watch",
    "Producers",
    "Licensors",
    "Ranked"
]

#lista para almacenar las columnas de score de 1 a 10 (score-1) ya que van a ser eliminadas
columnas_score = []
for columna in animeCSV.columns:
    if "score-" in columna.lower():
        columnas_score.append(columna)

# juntamos todas las listas que se quieren eliminar
columnasEliminar = columnas_eliminar + columnas_score

# eliminar columnas del csv
animeCSV = animeCSV.drop(*columnasEliminar)

# mostrar dataset para ver si se han eliminado correctamente
animeCSV.printSchema()
animeCSV.show(10)

# cambiar nombres con espacios (normalizar)
animeCSV = animeCSV.withColumnRenamed("English name", "English_name")

# print para indica cambio de tipo
print("Cambiar tipo de la columna Score (de String a Float)")

# conversion de tipos
animeCSV = animeCSV.replace("Unknown", None, subset=["Score"])
animeCSV = animeCSV.withColumn("Score", col("Score").cast("float"))
animeCSV.show(10)

# limpiar valores nulos (solo se realiza en la columna Score)
print("Valores nulos:")
animeCSV.select([count(when(col(c).isNull(), c)).alias(c) for c in animeCSV.columns]).show(truncate=False)

media_score = animeCSV.select(avg("Score")).first()[0]
print("media:", media_score)
animeCSV = animeCSV.fillna({"Score": media_score})

print("Valores nulos:")
animeCSV.select([count(when(col(c).isNull(), c)).alias(c) for c in animeCSV.columns]).show(truncate=False)

# En rating CSV hay que cambiar que rating sea double como en valoraciones
ratingCSV = ratingCSV.withColumn("rating", col("rating").cast(DoubleType()))
# Unir rating con las valoraciones del usuario
ratingsTotal = ratingCSV.unionByName(valoracionesCSV)

# modificar columna de generos para separación y tipos para tener todo en minusculas
animeCSV = animeCSV.withColumn("Genres", split(col("Genres"), ", "))
animeCSV = animeCSV.withColumn("Type", lower(col("Type")))

# Preparar dataset para ALS, cambio de nombres de columnas user_id y anime_id
ratingsALS = ratingsTotal.withColumn("rating", col("rating").cast("double")).withColumnRenamed("user_id", "userId").withColumnRenamed("anime_id", "itemId")
print("Info")
animeCSV.show(10)
ratingsALS.show(10)

# comprobacion union ratings y valoraciones
ratingsEP = ratingsALS.filter(col("userId") == "666666")
ratingsEP.show(2)

# crear columna para valoracion media por item
columnaValoracionMedia = ratingsALS.groupby("itemId").agg(avg("rating").alias("valoracion_media"))

animeCSV = animeCSV.join(columnaValoracionMedia, animeCSV.ID == columnaValoracionMedia.itemId, how="left").drop("itemId")
# Limpiar valores nulos en valoracion_media -> porque no todos los animes están valorados en ratings_complete.csv
media_valoracion = animeCSV.select(avg("valoracion_media")).first()[0]
print("Media:", media_valoracion)
animeCSV = animeCSV.fillna({"valoracion_media": media_valoracion})
animeCSV.show(10)
animeCSV.printSchema()

# ANALISIS EXPLORATORIO
print("\033[1mANALISIS EXPLORATORIO DE LOS DATOS\033[0m")

print("\033[1m-TOP 10 MEJOR VALORADOS\033[0m")
# Top 10 mejor valorados según Score y valoración media
print("Por Score")
animeCSV.orderBy(col("Score").desc()).select("Name", "English_name", "Score").show(10)
print("Por Valoracion media")
animeCSV.orderBy(col("valoracion_media").desc()).select("Name", "English_name", "valoracion_media").show(10)


print("\033[1m-TOP 10 PEOR VALORADOS\033[0m")
# Top 10 peor valorados según Score y valoración media
print("Por Score")
animeCSV.orderBy(col("Score").asc()).select("Name", "English_name", "Score").show(10)
print("Por valoracion media")
animeCSV.orderBy(col("valoracion_media").asc()).select("Name", "English_name", "valoracion_media").show(10)

# Relación entre género y valoraciones
# Convertir cada fila del array de generos en una columna diferente para poder obtener estadísticas por género
animeGenres = animeCSV.withColumn("genre", explode(col("Genres")))

# Media por género por Score
print("\033[1m-MEDIA POR GÉNERO\033[0m")
genreRatingsScore = animeGenres.groupBy("genre").agg(avg("Score").alias("media_score"), count("*").alias("Numero de animes")).orderBy(col("media_score").desc())
print("Por Score")
genreRatingsScore.show(10,truncate=False)

# Media por género por valoracion media
genreRatingsValoracion = animeGenres.groupBy("genre").agg(avg("valoracion_media").alias("media_valoracion"), count("*").alias("Numero de animes")).orderBy(col("media_valoracion").desc())
print("Por valoracion media")
genreRatingsValoracion.show(10, truncate=False)

print("\033[1m-FRECUENCIA DE ANIMES POR GÉNERO\033[0m")
df_mediaScore = animeGenres.groupBy("genre").agg(avg("Score").alias("media_score")).orderBy(col("media_score").desc()).toPandas()
print("Por Score")
plt.figure(figsize=(10,8))
plt.barh(df_mediaScore["genre"], df_mediaScore["media_score"])
plt.title("Valoración media por género")
plt.xlabel("Score medio")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

df_mediaScore = animeGenres.groupBy("genre").agg(avg("valoracion_media").alias("media_valoracion")).orderBy(col("media_valoracion").desc()).toPandas()
print("Por valoracion media")
plt.figure(figsize=(10,8))
plt.barh(df_mediaScore["genre"], df_mediaScore["media_valoracion"])
plt.title("Valoración media por género")
plt.xlabel("Valoracion media medio")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



# Estudios con mejor y peor nota media
studioRatings = animeCSV.withColumn("studio", explode(split(col("Studios"), ", ")))

studioRatingsScore = studioRatings.groupBy("studio").agg(avg("Score").alias("media_score"), count("*").alias("Numero de animes")).orderBy(col("media_score").desc())
studioRatingsValoracion = studioRatings.groupBy("studio").agg(avg("valoracion_media").alias("media_valoracion"), count("*").alias("Numero de animes")).orderBy(col("media_valoracion").desc())
# mejor valoracion 
print("\033[1m-ESTUDIOS CON MEJOR MEDIA\033[0m")
print("Por Score")
studioRatingsScore.show(10, truncate=False)  
print("Por valoracion media")
studioRatingsValoracion.show(10, truncate=False)  

# peor valoracion
print("\033[1m-ESTUDIOS CON PEOR MEDIA \033[0m")
print("Por Score")
studioRatingsScore.orderBy(col("media_score").asc()).show(10, truncate=False)  
print("Por valoracion media")
studioRatingsValoracion.orderBy(col("media_valoracion").asc()).show(10, truncate=False)  

# Histograma que muestra cómo se distribuyen las notas de valoracion de los animes
# por Score
print("\033[1m-HISTOGRAMA DE DISTRIBUCIÓN DE NOTAS\033[0m")
print("Por Score")
df = animeCSV.select("Score").toPandas()

plt.figure(figsize=(8,4))
plt.hist(df["Score"], bins=20)
plt.title("Distribución global")
plt.xlabel("Score")
plt.ylabel("Frecuencia")
plt.show()

#por valoracion media
df = animeCSV.select("valoracion_media").toPandas()
print("Por valoracion media")
plt.figure(figsize=(8,4))
plt.hist(df["valoracion_media"], bins=20)
plt.title("Distribución global")
plt.xlabel("Valoración media")
plt.ylabel("Frecuencia")
plt.show()


# Nota media general 
print("\033[1m-VALORACIONES MEDIAS\033[0m")
print("Score:", media_score)
print("Valoracion media:", media_valoracion)

# Moda
print("\033[1m-MODA (VALORES MÁS FRECUENTES)\033[0m")
print("Score")
moda = animeCSV.groupBy("Score").agg(count("*").alias("frecuencia_score")).orderBy(col("frecuencia_score").desc())
moda.show(5)

# por valoracion media
print("Valoracion media")
moda = animeCSV.groupBy("valoracion_media").agg(count("*").alias("frecuencia_valoracion_media")).orderBy(col("frecuencia_valoracion_media").desc())
moda.show(5)

# Comparación entre popularidad (número de miembros) y valoración para conocer si los más populares son los mejores
print("\033[1m-COMPARATIVA ENTRE POPULARIDAD Y VALORACIONES\033[0m")
print("Correlación")

correlacion = animeCSV.select(corr("Members", "Score")).first()[0]
print(" Members - Score:", correlacion)
correlacion = animeCSV.select(corr("Members", "valoracion_media")).first()[0]
print(" Members - Valoracion_media:", correlacion)

print("Grafico de popularidad y valoraciones")
# Pasar a Pandas
df_pop = animeCSV.select("Members", "Score", "valoracion_media").toPandas()

# Crear rangos de popularidad (10 grupos)
df_pop['Rango_popularidad'] = pd.qcut(df_pop['Members'], q=10)  

# Calcular el promedio de Score y valoración_media dentro de cada grupo
mean_scores = df_pop.groupby('Rango_popularidad')[['Score', 'valoracion_media']].mean().reset_index()

# Convertir los intervalos a string para Seaborn
mean_scores['Rango_popularidad'] = mean_scores['Rango_popularidad'].astype(str)

# Gráfico usando Seaborn
plt.figure(figsize=(12,6))
sns.lineplot(x='Rango_popularidad', y='Score', data=mean_scores, marker='o', label="Score promedio")
sns.lineplot(x='Rango_popularidad', y='valoracion_media', data=mean_scores, marker='o', label="Valoración media (usuarios)")

plt.xticks(rotation=45)
plt.title("Score vs Valoración Media según popularidad")
plt.ylabel("Valor promedio")
plt.xlabel("Rango de popularidad (Members)")
plt.legend()
plt.tight_layout()
plt.show()


# Análisis por tipo (TV, Movie, OVA, etc)
print("\033[1m-ANALISIS POR TYPE\033[0m")
print("Por Score")
typeRatingsScore = animeCSV.groupBy("Type").agg(avg("Score").alias("media_score"), count("*").alias("Numero de animes"))
typeRatingsScore.show()
print("Por valoracion media")
typeRatingsValoracion = animeCSV.groupBy("Type").agg(avg("valoracion_media").alias("media_valoracion"), count("*").alias("Numero de animes"))
typeRatingsValoracion.show()

print("Grafico de animes por tipo")
df = animeCSV.groupBy("Type").count().orderBy(col("count").desc()).toPandas()

plt.figure(figsize=(8,8))
plt.barh(df["Type"], df["count"])
plt.title("Frecuencia de animes por tipo")
plt.xlabel("Número de animes")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Analisis usuario EP
print("\033[1m-ANALISIS DEL USUARIO EP (USUARIO DE VALORACIONES_EP)\033[0m")
# filtrar para obtener solo las valoraciones de este usuario
ratingsEP = ratingsALS.filter(col("userId") == 666666)

# Media de EP
mediaEP = ratingsEP.select(avg("rating")).first()[0]
print("Nota media usuario EP:", mediaEP)

# Top géneros del usuario EP
print("Gráfico del top de géneros del usuario")
ratingsEP_genres = ratingsEP.join(animeGenres, ratingsEP.itemId == animeGenres.ID, "left")
ratingsEP_genres.groupBy("genre").agg(avg("rating").alias("media_valoracion"), count("*").alias("num_valoraciones")).orderBy(col("media_valoracion").desc()).show(10)

# Comparar los géneros preferido del usuario EP -> comparar la media de los generos con la media global de ese genero
# Se analiza los géneros preferidos del usuario EP comparando su media con la media global
df_usuario_EP = ratingsEP.join(animeGenres, ratingsEP.itemId == animeCSV.ID, "left").groupBy("genre").agg(avg("rating").alias("media_rating")).orderBy(col("media_rating").desc()).toPandas()

plt.figure(figsize=(8,8))
plt.barh(df_usuario_EP["genre"], df_usuario_EP["media_rating"])
plt.title("Géneros favoritos del usuario EP")
plt.xlabel("Valoración media")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Media global por género 
generos_globales = animeGenres.groupBy("genre").agg(avg("valoracion_media").alias("media_global")).toPandas()

# Media del usuario EP por género 
generos_EP = ratingsEP.join(animeGenres, ratingsEP.itemId == animeGenres.ID, "left").groupBy("genre").agg(avg("rating").alias("media_usuario")).toPandas()

# Unir ambas tablas
comparativa = pd.merge(generos_EP, generos_globales, on="genre", how="inner")

# Ordenar por diferencia para que se vea más claro
comparativa["diferencia"] = comparativa["media_usuario"] - comparativa["media_global"]
comparativa = comparativa.sort_values("diferencia", ascending=False)

print("Comparativa de géneros favoritos: Usuario EP vs Media global")
plt.figure(figsize=(10,6))
bar_width = 0.4
x = np.arange(len(comparativa))

plt.bar(x - bar_width/2, comparativa["media_usuario"], width=bar_width, label="Usuario EP", color='skyblue')
plt.bar(x + bar_width/2, comparativa["media_global"], width=bar_width, label="Media global", color='salmon')

plt.xticks(x, comparativa["genre"], rotation=45, ha='right')
plt.ylabel("Valoración media")
plt.title("Comparativa de géneros favoritos: Usuario EP vs Media global")
plt.legend()
plt.tight_layout()
plt.show()



# Relación entre Dropped y Completed -> comparación entre la cantidad de usuarios que han terminado el anime y quienes lo han abandonado 
print("\033[1m-RELACION ENTRE COMPLETED Y DROPPED\033[0m")
animeCSV.groupBy("Type").agg(sum("Dropped").alias("total_dropped"),sum("Completed").alias("total_completed"),
    (sum("Completed") / (sum("Completed") + sum("Dropped"))).alias("ratio_completed")).show()

# Crear dataframe con totales por Type
df_porcentaje = animeCSV.groupBy("Type").agg(sum("Dropped").alias("total_dropped"),sum("Completed").alias("total_completed")).toPandas()

# Calcular porcentajes
df_porcentaje["pct_completed"] = df_porcentaje["total_completed"] / (df_porcentaje["total_completed"] + df_porcentaje["total_dropped"])
df_porcentaje["pct_dropped"] = df_porcentaje["total_dropped"] / (df_porcentaje["total_completed"] + df_porcentaje["total_dropped"])

# Gráfico stacked bar con porcentajes
types = df_porcentaje["Type"]
x = np.arange(len(types))

plt.figure(figsize=(10,6))

plt.bar(x, df_porcentaje["pct_completed"], label="Completed")
plt.bar(x, df_porcentaje["pct_dropped"], bottom=df_porcentaje["pct_completed"], label="Dropped")

plt.xticks(x, types, rotation=45)
plt.ylabel("Porcentaje")
plt.title("Porcentaje de Completed vs Dropped por tipo de anime")

# Mostrar porcentajes en eje Y de 0 a 1 como 0–100%
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.legend()
plt.tight_layout()
plt.show()

# ALGORITMO ALS
# Entrenamiento
training, test = ratingsALS.randomSplit([0.8, 0.2])
als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="itemId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(training)

# Predicción
predictions = model.transform(test)
rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction").evaluate(predictions)
print("RMSE =", rmse)

# Recomendaciones usuario 666666
idUsuario = 666666
user_df = spark.createDataFrame([(idUsuario,)], ["userId"])
recs = model.recommendForUserSubset(user_df, 50)
recs_final = recs.select(explode(col("recommendations")).alias("rec")).select(col("rec.itemId").alias("anime_id"), col("rec.rating").alias("predicted_rating")).join(animeCSV, col("anime_id") == animeCSV.ID, "inner").select(col("anime_id"), col("Name").alias("titulo_original"), col("English_name").alias("titulo_ingles"), col("Type"), col("valoracion_media")).filter(col("Type").isin("movie","tv")).orderBy(col("valoracion_media").desc())

# Guardar recomendaciones en .txt
ruta_base = "/scripts/recomendaciones_usuario_666666"
for tipo in ["movie", "tv"]:
    ruta_tipo = f"{ruta_base}/{tipo}"
    (
        recs_final
        .filter(col("Type") == tipo)
        .limit(5)
        .select(concat_ws(" | ", col("anime_id"), col("titulo_original"), col("titulo_ingles"), col("valoracion_media")).alias("value"))
        .coalesce(1)
        .write
        .mode("overwrite")
        .text(ruta_tipo)
    )

# Renombrar el archivo .txt part-xxxxx.txt → recomendaciones.txt
    for file in os.listdir(ruta_tipo):
        if file.startswith("part-") and file.endswith(".txt"):
            os.rename(os.path.join(ruta_tipo, file), os.path.join(ruta_tipo, "recomendaciones.txt"))

print("Fin del Algoritmo")

