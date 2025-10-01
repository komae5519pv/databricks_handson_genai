# Databricks notebook source
# MAGIC %md
# MAGIC **要件: DBR 16.4 ML（シングルノード）以降を使用してください**<br>
# MAGIC
# MAGIC - データはクラスタのドライバーノード上に保存して処理します。
# MAGIC - クラスタを**シングルノード（ワーカーなし）**で動かしてください。
# MAGIC - マルチノードやオートスケール構成だと、ワーカーノードからローカルファイルが見えず、エラーになります。
# MAGIC
# MAGIC [基盤モデル ファインチューニング APIを使用してトレーニング 実行を作成します](https://docs.databricks.com/aws/ja/large-language-models/foundation-model-training/create-fine-tune-run#configure)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 初期設定

# COMMAND ----------

# MAGIC %pip install databricks_genai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

tutorial_path = "/databricks/driver" # 
import os
os.environ['TUTORIAL_PATH']=tutorial_path # 後ほどShellコマンドからアクセスするため環境変数にセット

# 比較用のベースモデル
base_model_endpoint_name = "databricks-meta-llama-3-1-8b-instruct"

# ファインチューニング用の元LLM
model = 'meta-llama/Llama-3.2-3B-Instruct'

# ファインチューニング後のモデル名
ft_model_endpoint_name = "ift-meta-llama-3-1-8b-instruct"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Livedoorニュースコーパス内の全記事をダウンロード
# MAGIC
# MAGIC Databricks クラスタードライバーノード上で直接シェルスクリプトを実行し、
# MAGIC Livedoorニュースコーパス（ニュース記事）データを取得
# MAGIC
# MAGIC [株式会社ロンウイットが提供するデータセット: Livedoorニュースコーパス](https://www.rondhuit.com/download.html#news%20corpus)

# COMMAND ----------

# MAGIC %sh
# MAGIC cd $TUTORIAL_PATH
# MAGIC
# MAGIC # データのダウンロード
# MAGIC wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
# MAGIC
# MAGIC # ファイルの解凍
# MAGIC tar -zxf ldcc-20140209.tar.gz 

# COMMAND ----------

# MAGIC %sh
# MAGIC ls $TUTORIAL_PATH/text

# COMMAND ----------

# MAGIC %sh
# MAGIC ls $TUTORIAL_PATH/text/it-life-hack/

# COMMAND ----------

# MAGIC %sh
# MAGIC cat $TUTORIAL_PATH/text/it-life-hack/it-life-hack-6342280.txt

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Livedoorニュースコーパス内の全記事を一覧化（どの記事がどのカテゴリに属し、どこのファイルパスに保存されているか）
# MAGIC 次のコードでLivedoorニュース記事を
# MAGIC - ラベル番号（label）
# MAGIC - カテゴリ名（label_name）
# MAGIC - 各記事ファイルのパス（file_path）
# MAGIC
# MAGIC という3つの情報でリスト化（pandasのデータフレーム化）します。

# COMMAND ----------

import glob
import pandas as pd

# カテゴリーのリスト
category_list = [
    'dokujo-tsushin',
    'it-life-hack',
    'kaden-channel',
    'livedoor-homme',
    'movie-enter',
    'peachy',
    'smax',
    'sports-watch',
    'topic-news'
]

# 各データの形式を整える
columns = ['label', 'label_name', 'file_path']
dataset_label_text = pd.DataFrame(columns=columns)
id2label = {} 
label2id = {}
for label, category in enumerate(category_list):
  
  file_names_list = sorted(glob.glob(f'{tutorial_path}/text/{category}/{category}*'))#対象メディアの記事が保存されているファイルのlistを取得します。
  print(f"{category}の記事を処理しています。　{category}に対応する番号は{label}で、データ個数は{len(file_names_list)}です。")

  id2label[label] = category
  label2id[category] = label
  
  for file in file_names_list:
      list = [[label, category, file]]
      df_append = pd.DataFrame(data=list, columns=columns)
      dataset_label_text = pd.concat([dataset_label_text, df_append], ignore_index=True, axis=0)


dataset_label_text.head()

# COMMAND ----------

display(dataset_label_text)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

@pandas_udf(StringType())
def read_text(paths: pd.Series) -> pd.Series:

  all_text = []
  for index, file in paths.items():             #取得したlistに従って実際のFileからデータを取得します。
      lines = open(file).read().splitlines()
      text = '\n'.join(lines[3:])               # ファイルの4行目からを抜き出す。
      all_text.append(text)

  return pd.Series(all_text)

# COMMAND ----------

from pyspark.sql.functions import col

dataset_df = spark.createDataFrame(dataset_label_text)
dataset_df = dataset_df.withColumn('text', read_text(col('file_path')))
display(dataset_df.head(5))

# COMMAND ----------

# DBTITLE 1,ニュース記事データセットをテーブル保存
spark.sql(f"""
CREATE CATALOG IF NOT EXISTS {catalog_name};
""")

spark.sql(f"""
CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name};
""")

dataset_df.write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.ldcc_source")

# COMMAND ----------

# MAGIC %md
# MAGIC ![catalog_dataset.png](./catalog_dataset.png "catalog_dataset.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. ファインチューニング用のデータ準備

# COMMAND ----------

# DBTITLE 1,トレーニング用とテスト用のデータに分ける
# ソーステーブルを読み込む
source_df = spark.table(f"{catalog_name}.{schema_name}.ldcc_source")

# データをトレーニング用とテスト用に分割する
train_df, test_df = source_df.randomSplit([0.95, 0.05], seed=42)

# トレーニング用とテスト用の一時ビューを作成する
train_df.createOrReplaceTempView("train_temp_view")
test_df.createOrReplaceTempView("test_temp_view")

# トレーニング用とテスト用のテーブルを作成する
train_df.write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.training_table")
test_df.write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.test_table")

# COMMAND ----------

# MAGIC %md
# MAGIC ![catalog_train_test.png](./catalog_train_test.png "catalog_train_test.png")

# COMMAND ----------

# プロンプトはもっと改善した方が良いと思います
system_prompt = """
下記はニュース記事のカテゴリとその内容の一覧です。
dokujo-tsushin: 独身女性向けの記事
it-life-hack: ITライフハックに関する記事
kaden-channel: 家電に関する記事
livedoor-homme: 20代後半から40代男性をターゲットにしたウェブマガジン
movie-enter: 映画に関する記事
peachy: 毎日をハッピーに生きる女性のためのニュースサイト
smax: スマートフォンを中心に、モバイル関連情報をお届けするお役立ちサイト
sports-watch: スポーツのニュース
topic-news: トピックニュース

下記のニュース記事の内容をもとに、[dokujo-tsushin,it-life-hack,kaden-channel,livedoor-homme,movie-enter,peachy,smax,sports-watch,topic-news]のいずれかのカテゴリに分類してください。
回答には分類したカテゴリを1つのみを含め、補足説明などそれ以外の一切の単語は含めないでください。:\n\n"""

# COMMAND ----------

# DBTITLE 1,素のLlamaで検証
spark.sql(f"""
            create or replace table {catalog_name}.{schema_name}.classified_by_llm AS
            SELECT
            text,
            ai_query('{base_model_endpoint_name}', concat('{system_prompt}', text)) AS base_model_classification,
            label_name AS correct_classification
        FROM {catalog_name}.{schema_name}.test_table
""").display()

display(spark.table(f"{catalog_name}.{schema_name}.classified_by_llm"))

# COMMAND ----------

# MAGIC %md
# MAGIC ![catalog_classified_by_llm.png](./catalog_classified_by_llm.png "catalog_classified_by_llm.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. UIでファインチューニング
# MAGIC ここからFine Tuning
# MAGIC
# MAGIC まずはトレーニング用のデータセットを作成

# COMMAND ----------

# DBTITLE 1,トレーニングJSONLファイル保存（Volume）
# # SQLを使用して目的のデータ構造を持つ DataFrame を作成
# finetuning_df = spark.sql(f"""
# SELECT 
#     ARRAY(
#         STRUCT('user' AS role, CONCAT("{system_prompt}", '\n', text) AS content),
#         STRUCT('assistant' AS role, label_name AS content)
#     ) AS messages
# FROM {catalog_name}.{schema_name}.training_table
# """)

# # ボリューム名とファイル名を定義
# volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/train/train.jsonl"

# # JSONL形式でUCボリュームに書き出す
# finetuning_df.coalesce(1).toPandas().to_json(
#     volume_path,
#     orient='records',
#     lines=True,
#     force_ascii=False # 日本語などのマルチバイト文字をそのまま保存
# )

# print(f"トレーニングデータはUCボリュームに保存されました: {volume_path}")

# COMMAND ----------

# DBTITLE 1,トレーニングテーブル保存（UC）
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.finetuning_training_table AS
SELECT 
    ARRAY(
        STRUCT('user' AS role, CONCAT("{system_prompt}", '\n', text) AS content),
        STRUCT('assistant' AS role, label_name AS content)
    ) AS messages
FROM {catalog_name}.{schema_name}.training_table;
""")

spark.table(f"{catalog_name}.{schema_name}.finetuning_training_table").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ![catalog_finetune_train_data.gif](./catalog_finetune_train_data.gif "catalog_finetune_train_data.gif")

# COMMAND ----------

# DBTITLE 1,Option: 評価用データ作成
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.finetuning_evaluation_table AS
SELECT 
    ARRAY(
        STRUCT('user' AS role, CONCAT("{system_prompt}", '\n', text) AS content),
        STRUCT('assistant' AS role, label_name AS content)
    ) AS messages
FROM {catalog_name}.{schema_name}.test_table;
""")

spark.table(f"{catalog_name}.{schema_name}.finetuning_evaluation_table").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ![catalog_finetune_eval_data.gif](./catalog_finetune_eval_data.gif "catalog_finetune_eval_data.gif")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ファインチューニングの実施
# MAGIC UIを利用
# MAGIC
# MAGIC 3epoch で21.5min
# MAGIC
# MAGIC ![model_metrics_3ep.png](./model_metrics_3ep.png "model_metrics_3ep.png")
# MAGIC
# MAGIC 5epoch で29.6min
# MAGIC
# MAGIC ![model_metrics_5ep.png](./model_metrics_5ep.png "model_metrics_5ep.png")
# MAGIC
# MAGIC 10epoch で45.9min
# MAGIC
# MAGIC ![model_metrics_10ep.png](./model_metrics_10ep.png "model_metrics_10ep.png")

# COMMAND ----------

# テーブルの読み込みとJSONLへの変換で使われるクラスタID
data_prep_cluster_id = "1001-050207-9wpq3x4c"

# COMMAND ----------

spark.sql(f"GRANT USE CATALOG ON CATALOG {catalog_name} TO `account users`;")
spark.sql(f"GRANT CREATE MODEL ON CATALOG {catalog_name} TO `account users`;")
spark.sql(f"GRANT USE SCHEMA ON SCHEMA {catalog_name}.{schema_name} TO `account users`;")
spark.sql(f"GRANT ALL PRIVILEGES ON SCHEMA {catalog_name}.{schema_name} TO `account users`;")
spark.sql(f"GRANT CREATE MODEL ON SCHEMA {catalog_name}.{schema_name} TO `account users`;")

# COMMAND ----------

# DBTITLE 1,トレーニング実行
from databricks.model_training import foundation_model as fm

# ハイパーパラメーター
learningRates = ["5e-7", "1e-6", "5e-6", "1e-5"]
epocs = ['3ep','5ep','10ep']

# エクスペリメント設定
current_username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
run_name = f"llama-3b-fine-tune"

run_ids = []  # run_id を格納する配列

# ファインチューニング実行（トレーニングデータとしてUCテーブルを指定）
for ep in epocs:
  for lr in learningRates:
    run = fm.create(
      model= model,
      train_data_path=f"{catalog_name}.{schema_name}.finetuning_training_table", # トレーニングデータ@UCテーブル
      # Public HF dataset is also supported
      # train_data_path='mosaicml/dolly_hhrlhf/train'
      task_type = "CHAT_COMPLETION",
      experiment_path = f"/Users/{current_username}/{run_name}",
      # register_to=f"{catalog_name}.{schema_name}", # UC登録するモデルパス(モデル名自動採番)
      register_to=f"{catalog_name}.{schema_name}.{ft_model_endpoint_name}-{ep}-{lr}", # UC登録するモデルパス
      learning_rate = lr,
      training_duration = ep,
      data_prep_cluster_id = data_prep_cluster_id   # UCテーブルデータの読込み＆JSONL変換のためのクラスタID
    )
    run_id = run.run_id
    run_ids.append(run_id)  # 配列に追加

# # ファインチューニング実行（トレーニングデータとしてVolume上のJSONLファイルを指定）
# for ep in epocs:
#   for lr in learningRates:
#     run = fm.create(
#       model= model,
#       train_data_path=f"dbfs:/Volumes/{catalog_name}/{schema_name}/{volume_name}/train/train.jsonl", # UC Volume with JSONL formatted data
#       # Public HF dataset is also supported
#       # train_data_path='mosaicml/dolly_hhrlhf/train'
#       task_type = "CHAT_COMPLETION",
#       experiment_path = f"/Users/{current_username}/{run_name}",
#       # register_to=f"{catalog_name}.{schema_name}", # UC登録するモデルパス(モデル名自動採番)
#       register_to=f"{catalog_name}.{schema_name}.{ft_model_endpoint_name}-{ep}-{lr}", # UC登録するモデルパス
#       learning_rate = lr,
#       training_duration = ep
#     )
#     run_id = run.run_id
#     run_ids.append(run_id)  # 配列に追加

print(run_ids)  # すべてのrun_idを出力

# COMMAND ----------

# DBTITLE 1,UCモデルにエイリアスをつける
import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

# MLflowレジストリの設定
mlflow.set_registry_uri('databricks-uc')
mlflow_client = MlflowClient()
sdk_client = WorkspaceClient()

# モデル名・バージョンごとにUCエイリアス設定と権限付与
for ep in epocs:
    for lr in learningRates:
        MODEL_NAME = f"{catalog_name}.{schema_name}.{ft_model_endpoint_name}-{ep}-{lr}"

        # バージョン情報の一覧を取得
        model_versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
        if not model_versions:
            print(f"{MODEL_NAME}はまだ登録されていません。")
            continue
        # 最新バージョンを取得
        latest_model_version = max([int(m.version) for m in model_versions])
        print(f"{MODEL_NAME}の最新バージョン: {latest_model_version}")

        # UC エイリアスを"prod"に設定
        mlflow_client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="prod",
            version=latest_model_version
        )

        # 全ユーザー(グループ名: account users)にモデルの権限を付与
        sdk_client.grants.update(
            c.SecurableType.FUNCTION,
            MODEL_NAME,
            changes=[c.PermissionsChange(add=[c.Privilege["ALL_PRIVILEGES"]], principal="account users")]
        )


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. モデルのサービング
# MAGIC UIで実施
# MAGIC 利用可能までおそらく5-6分くらい<br>
# MAGIC <!-- モデルサービングエンドポイント名: `ift-meta-llama-3-1-8b-instruct` -->
# MAGIC モデルサービングエンドポイント名: [ift-meta-llama-3-1-8b-instruct-5ep-5e-6](https://e2-demo-field-eng.cloud.databricks.com/ml/endpoints/ift-meta-llama-3-1-8b-instruct-5ep-5e-6?o=1444828305810485)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. 検証
# MAGIC ファインチューニングしたモデルで分類してみる

# COMMAND ----------

spark.sql(f"""
            create or replace table {catalog_name}.{schema_name}.classified_by_finetuned AS
            SELECT
                text,
                ai_query(
                    -- "ift-meta-llama-3-1-8b-instruct",
                    "ift-meta-llama-3-1-8b-instruct-5ep-5e-6",
                    concat("{system_prompt}", text)
                ) AS base_model_classification,
            label_name AS correct_classification
        FROM {catalog_name}.{schema_name}.test_table
""").display()
display(spark.table(f"{catalog_name}.{schema_name}.classified_by_finetuned"))

# COMMAND ----------

# MAGIC %md
# MAGIC [ダッシュボード](https://e2-demo-field-eng.cloud.databricks.com/dashboardsv3/01f09ecab24d1104858963b359365290/published?o=1444828305810485)で結果を比較してみる
