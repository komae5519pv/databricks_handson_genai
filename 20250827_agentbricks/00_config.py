# Databricks notebook source
# カタログ、スキーマ、ボリューム
MY_CATALOG = "komae_demo_v4"             # 使用したいカタログ名に変更してください
MY_SCHEMA = "demo_agentbricks"
MY_VOLUME = "raw_data"

# COMMAND ----------

# カタログ、スキーマ、ボリューム作成
spark.sql(f"CREATE CATALOG IF NOT EXISTS {MY_CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {MY_CATALOG}.{MY_SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {MY_CATALOG}.{MY_SCHEMA}.{MY_VOLUME}")

spark.sql(f"USE CATALOG {MY_CATALOG}")
spark.sql(f"USE SCHEMA {MY_SCHEMA}")

# ボリュームのサブディレクトリ作成
dbutils.fs.mkdirs(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/image")
dbutils.fs.mkdirs(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/pdf")
dbutils.fs.mkdirs(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/csv")

print(f"MY_CATALOG: {MY_CATALOG}")
print(f"MY_SCHEMA: {MY_SCHEMA}")
print(f"MY_VOLUME: {MY_VOLUME}/image")
print(f"MY_VOLUME: {MY_VOLUME}/pdf")
print(f"MY_VOLUME: {MY_VOLUME}/csv")
