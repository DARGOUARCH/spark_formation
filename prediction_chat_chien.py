#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from PIL import Image
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.pipeline import PipelineModel

# ----------------------------------------------------------------------------
# Arguments CLI
# ----------------------------------------------------------------------------
def parse_args():
      parser = argparse.ArgumentParser(
            description="Prédiction Cats / Dogs avec un modèle Spark ML existant."
      )
      parser.add_argument(
            "--input",
            required=True,
            help="Chemin des images à prédire (un dossier ou un fichier unique)"
      )
      parser.add_argument(
            "--model-path",
            required=True,
            help="Répertoire du modèle Spark ML sauvegardé (PipelineModel)"
      )
      parser.add_argument(
            "--image-size",
            type=int,
            default=64,
            help="Taille utilisée pour redimensionner les images"
      )
      return parser.parse_args()               
  
# ----------------------------------------------------------------------------
# Spark session
# ----------------------------------------------------------------------------
def build_spark():
      return (
            SparkSession.builder
            .appName("Prédiction_Chat_Chien")
            .getOrCreate()
      )
          
# ----------------------------------------------------------------------------
# Charger images et produire features
# ----------------------------------------------------------------------------
# 
def make_image_to_vec_udf(image_size):
      target = (image_size, image_size)
      def _extract(image):
            try:
                  raw = bytes(image.data)                  
                  base_mode = "RGB" if image.nChannels == 3 else "L"
                  img = Image.frombytes(base_mode, (image.width, image.height), raw)
                  # Forcer grayscale -> 100*100 = 10 000 features si image_size=100
                  img = img.convert("L")
                  img = img.resize(target)
                  arr = np.asarray(img, dtype="float32") / 255.0
                  return arr.reshape(-1).tolist()
            except Exception:
                  return None
      return udf(_extract, ArrayType(DoubleType()))

def load_images_and_vectorize(spark, input_path, image_size):
      df = spark.read.format("image").load(input_path)
      # image → array<float>
      to_vec = make_image_to_vec_udf(image_size)
      df = df.withColumn("features_array", to_vec(col("image")))
      df = df.filter(col("features_array").isNotNull())
      # array -> Vector
      arr_to_vec = udf(lambda x: Vectors.dense(x), VectorUDT())
      df = df.withColumn("features", arr_to_vec(col("features_array")))
      return df

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def predict(spark, model_path, df):
      print("Chargement du modèle Spark ML…")
      model = PipelineModel.load(model_path)
      print("Application du modèle…")
      preds = model.transform(df)
      # Décodage label 0->cats / 1->dogs
      def decode_label(l):
            return "cats" if l == 0.0 else "dogs"
      decode_udf = udf(decode_label)
      preds = preds.withColumn("prediction_label", decode_udf(col("prediction")))
      return preds

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
      args = parse_args()
      spark = build_spark()
      df = load_images_and_vectorize(
            spark,
            args.input,
            args.image_size
      )
      preds = predict(spark, args.model_path, df)
      print("\n=== Résultats ===")
      preds.select(
            col("image.origin"),
            col("prediction_label"),
            col("probability")
      ).show(truncate=False)
      spark.stop()

if __name__ == "__main__":
      main()
          