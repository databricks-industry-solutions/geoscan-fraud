# Databricks notebook source
# MAGIC %md
# MAGIC # Fraud prevention
# MAGIC In the previous [notebook]($./00_geofraud_context), we demonstrated how GEOSCAN can help financial services institutions leverage their entire dataset to better understand customers specific behaviours. In this notebook, we want to use the insights we have gained earlier to extract anomalous events and bridge the technological gap that exists between analytics and operations environments. More often than not, Fraud detection frameworks run outside of an analytics environment due to the combination of data sensitivity (PII), regulatory requirements (PCI/DSS) and model materiality (high SLAs and low latency). For these reasons, we explore here multiple strategies to serve our insights either as a self contained framework or through an online datastore (such as [redis](https://redis.io/), [mongodb](https://www.mongodb.com/) or [elasticache](https://aws.amazon.com/elasticache/) - although many other solutions may be viable)

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transactional context
# MAGIC As we've trained personalized models for each customer, we can easily understand the type of transactions as well as the day and hours these transactions usually take place. Are these clusters more "active" during working hours or on week ends? Are these transactions more about fast foods and coffee shops or are they driving fewer but more expensives items? Such a geospatial analytics framework combined with transaction enrichment (future solution accelerator) could tell us great information about our customers' spends beyond demographics, moving towards a customer centric approach to retail banking. Unfortunately, our synthetic dataset does not contain any additional attributes to learn behavioral pattern from. For the purpose of this exercise, we will retrieve our clusters (as tiled with H3 polygon as introduced earlier) as-is to detect transactions that happened outside of any known location. 

# COMMAND ----------

tiles = spark.read.table(config['database']['tables']['tiles'])
display(tiles)

# COMMAND ----------

# MAGIC %md
# MAGIC As the core of our framework relies on open data standards ([RFC7946](https://tools.ietf.org/html/rfc7946)), we could load our models as a simple Dataframe without relying on the GEOSCAN library. We simply have to read the `data` directory of our model output.

# COMMAND ----------

model_path = config['model']['path']
model_personalized = spark.read.format('parquet').load('{}/data'.format(model_path))
display(model_personalized)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extracting anomalies
# MAGIC Our (simplisitic) approach will be to detect if a transaction was executed in a popular area for each of our customers. Since we have stored and indexed all of our models as H3 tiles, it becomes easy to enrich each transaction with their cluster using a simple JOIN operation (for large scale processing) or lookup (for real time scoring) instead of complex geospatial queries like point in polygon search. Although we are using the H3 python API instead of GEOSCAN library, our generated H3 hexadecimal values are consistent - assuming we select the same resolution we used to generate those tiles (10). For reference, please have a look at the H3 [resolution table](https://h3geo.org/docs/core-library/restable)

# COMMAND ----------

from utils.spark_utils import *

# COMMAND ----------

# MAGIC %md
# MAGIC In the example below, we can easily extract  transactions happenning outside of any customer prefered locations. Please note that we previously relaxed our conditions by adding 3 extra layers of H3 polygons to capture transactions happenning in close vicinity of spending clusters

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F
transactions = pd.read_csv('data/transactions.csv')
transactions['latitude'] = transactions['latitude'].apply(lambda x: float(x))
transactions['longitude'] = transactions['longitude'].apply(lambda x: float(x))
transactions['amount'] = transactions['amount'].apply(lambda x: float(x))
points_df = spark.createDataFrame(transactions)
display(points_df)

# COMMAND ----------

from pyspark.sql import functions as F

anomalous_transactions = (
  points_df
    .withColumn('h3', to_h3(F.col('latitude'), F.col('longitude'), F.lit(10)))
    .join(tiles, ['user', 'h3'], 'left_outer')
    .filter(F.expr('cluster IS NULL'))
    .drop('h3', 'cluster', 'tf_idf')
)

display(anomalous_transactions)

# COMMAND ----------

# MAGIC %md
# MAGIC Out of half a million transactions, we extracted 81 records in less than 5 seconds. Not necessarily fraudulent, maybe not even suspicious, these transactions did not match any of our users "normal" behaviours, and as such, are worth flagging as part of an overhatching fraud prevention framework. In real life example, we should factor for time and additional transactional context. Would a same transaction happening on a Sunday afternoon or a Wednesday morning be suspicious given user characteristics we could learn? 

# COMMAND ----------

# MAGIC %md
# MAGIC Before moving forwards, it is always benefitial to validate our strategy (altough not empirically) using a simple visualization for a given customer (`99407ef8-40ae-424e-b9ae-9fd2e4838ec3`), reporting card transactions happenning outside of any known patterns.

# COMMAND ----------

import folium
from folium import plugins
from pyspark.sql import functions as F

user = '9cafdb6d-9134-4ee8-bdf6-972ebc3af729'
anomalies = anomalous_transactions.filter(F.col('user') == user).toPandas()
clusters = model_personalized.filter(F.col('user') == user).toPandas().cluster.iloc[0]

personalized = folium.Map([40.75466940037548,-73.98365020751953], zoom_start=12, width='80%', height='100%')
folium.TileLayer('Stamen Toner').add_to(personalized)

for i, point in list(anomalies.iterrows())[0:5]:
  folium.Marker([point.latitude, point.longitude], popup=point.amount).add_to(personalized)

folium.GeoJson(clusters, name="geojson").add_to(personalized)
personalized

# COMMAND ----------

# MAGIC %md
# MAGIC Although this synthetic data does not show evidence of suspicious transactions, we demonstrated how anomalous records can easily be extracted from a massive dataset without the need to run complex geospatial queries. In fact, the same can now be achieved using standard SQL functionalities in a notebook or in a SQL analytics workspace. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real time fraud detection
# MAGIC With millions of transactions and low latency requirements, it would not be realistic to join datasets in real time. Although we could load all clusters (their H3 tiles) in memory, we may have evaluated multiple models at different time of the days for different users, for different segments or different transaction indicators (e.g. for different brand category or [MCC codes](https://en.wikipedia.org/wiki/Merchant_category_code)) resulting in a complex system that requires efficient lookup strategies against multiple variables. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bloom filters
# MAGIC Here comes [Bloom Filters](https://en.wikipedia.org/wiki/Bloom_filter), an efficient probabilistic data structure than can test the existence of a given record without keeping an entire set in memory. Although bloom filters have been around for a long time, its usage has not - sadly - been democratized beyond complex engineering techniques such as database optimization engines and daunting execution planners (Delta engine leverages bloom filters optimizations among other techniques). This technique is a powerful tool worth having in a modern data science toolkit. 

# COMMAND ----------

# MAGIC %md
# MAGIC The concept behind a bloom filter is to convert a series of records (in our case a H3 location) into a series of hash values, overlaying each of their byte arrays representations as vectors of 1 and 0. Testing the existence of a given record results in testing the existence of each of its bits set to 1. Given a record `w`, if any of its bit is not found in our set, we are 100% sure we haven't seen record `w` before. However, it all of its bits are found in our set, it could be caused by an unfortunate succession of hash collisions. In other words, Bloom filters offer a false negative rate of 0 but a non zero false positive rate (records we wrongly assume have been seen) that can controlled by the length of our array and the number of hash functions.
# MAGIC 
# MAGIC <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Bloom_filter.svg/720px-Bloom_filter.svg.png">
# MAGIC 
# MAGIC [Source](https://en.wikipedia.org/wiki/Bloom_filter)

# COMMAND ----------

# MAGIC %md
# MAGIC We will be using the `pybloomfilter` python library to validate this approach, training a Bloom filter against each and every known H3 tile of a given user. Although our filter may logically contains millions of records, we would only need to physically maintain 1 byte array in memory to enable a probabilistic search (controlled here with a 1% false positive rate).

# COMMAND ----------

from utils.bloom_utils import *
records = list(tiles.filter(F.col('user') == user).toPandas().h3)
cluster = train_bloom_filter(records)

# COMMAND ----------

# MAGIC %md
# MAGIC We retrieve all the points we know exist and test our false negative rate (should be null)

# COMMAND ----------

normal_df = tiles.filter(F.col('user') == user).select(F.col('h3')).toPandas()
normal_df['matched'] = normal_df['h3'].apply(lambda x: x in cluster)
display(normal_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Similarly, we access our anomalous transactions to validate our false positive rate (should be lower than 1%). 

# COMMAND ----------

abnormal_df = (
  anomalous_transactions
    .filter(F.col('user') == user)
    .withColumn('h3', to_h3(F.col('latitude'), F.col('longitude'), F.lit(10)))
    .select('h3').toPandas()
)

abnormal_df['matched'] = abnormal_df['h3'].apply(lambda x: x in cluster)
display(abnormal_df)

# COMMAND ----------

# MAGIC %md
# MAGIC With our approach validated, we could build one bloom filter for each user, storing an "model" as a simple dictionary of users <> byte array

# COMMAND ----------

user_df = tiles.groupBy('user').agg(F.collect_list('h3').alias('tiles')).toPandas()
user_clusters = {}

for i, rec in user_df.iterrows():
  bf = train_bloom_filter(list(rec.tiles))
  user_clusters[rec.user] = bf

# COMMAND ----------

# MAGIC %md
# MAGIC We now have an efficient datastructure that can be used for real time lookup without having to maintain millions of H3 tiles in memory. For a given a transaction, we convert `latitude` and `longitude` to a H3 polygon (of size 10) and query the bloom filter for that specific user. Does that card transaction happenned in a familiar place?

# COMMAND ----------

'8A2A107252A7FFF' in user_clusters[user]

# COMMAND ----------

'9A2A1072C077FFF' in user_clusters[user]

# COMMAND ----------

# MAGIC %md
# MAGIC Using a `mlflow.pyFunc` pattern, we can wrap our business logic as a self packaged module that can be served real time, on stream, on SQL, or on demand, just like any ML / AI project. We just have to persist our data to disk to pass it onto our model and load bloom filters at model startup.

# COMMAND ----------

_ = (
  tiles
    .groupBy('user')
    .agg(F.collect_list('h3').alias('tiles'))
    .toPandas()
    .to_csv('{}/geoscan_tiles'.format(temp_directory))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Our logic expects a dataframe of `user`, `latitude` or `longitude` as an input, appending our records with `0` or `1` (whether we have observed this location before or not). 

# COMMAND ----------

# MAGIC %md
# MAGIC We ensure our dependencies (`pybloomfiltermmap3` and `h3`) are added to MLFlow conda environment. 

# COMMAND ----------

with mlflow.start_run(run_name='h3_lookup'):

  conda_env = mlflow.pyfunc.get_default_conda_env()
  conda_env['dependencies'][2]['pip'] += ['pybloomfiltermmap3=={}'.format(bloom_version())]
  conda_env['dependencies'][2]['pip'] += ['h3=={}'.format(h3.__version__)]
  
  artifacts = {
    'tiles': '{}/geoscan_tiles'.format(temp_directory),
  }
  
  mlflow.pyfunc.log_model(
    'pipeline', 
    python_model=H3Lookup(), 
    conda_env=conda_env,
    artifacts=artifacts
    )
  
  api_run_id = mlflow.active_run().info.run_id
  print(api_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model inference
# MAGIC With our model available as a `mlflow.pyfunc`, we can serve it from many different places, as a batch or on a stream, behind a custom API or a proprietary system, back on premises or using cloud based technologies. For more information about MLFlow deployment, please refer to documentation [here](https://www.mlflow.org/docs/latest/python_api/index.html). 

# COMMAND ----------

import mlflow
model = mlflow.pyfunc.load_model('runs:/{}/pipeline'.format(api_run_id))

# COMMAND ----------

transactions = points_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Out of the 81 transactions previously reported as suspicious, our bloom filters detected 78 of them (within our false positive rate) with a model that can be now executed real time in a third party environment.

# COMMAND ----------

anomalies = model.predict(transactions)
anomalies = anomalies[anomalies['anomaly'] != 0]
display(anomalies)

# COMMAND ----------

# MAGIC %md
# MAGIC However, this approach pauses an important operating challenge for large financial services organizations as new models would need to be constantly retrained and redeployed to adapt to users changing behaviours. Let's take an example a user going on holidays. Although their first card transactions may be returned as anomalous (not necessarily suspicious), such a strategy would need to adapt and learn the new "normal" as more and more transactions are observed. One would need to run the same process with new data, resulting in a new version of a model being released, reviewed by an independant team of experts, approved by a governance entity and eventually updated to a fraud production endpoints outside of any change freeze. Technically possible when supported by a strong operating model (data driven organizations), this approach may not be viable for many.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Online datastore
# MAGIC It is fairly common for financial services institutions to have an online data store decoupled from an analytics platform. A flow of incoming transactions are compared with reference data points in real time. An alternative approach to the above is to use an online datastore (like mongodb) to keep "pushing" reference data points to a live endpoint as a business as usual process (hence outside of ITSM change windows). Any incoming transaction would be matched against a set of rules constantly updated (reference data) and accessible via sub-seconds look up queries. Using [mongo db connector](https://docs.mongodb.com/spark-connector/current/) (as an example), we show how organizations can save our geo clusters dataframes for real time serving, combining the predictive power of advanced analytics with low latency of traditional rule based systems.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's first create a new collection (i.e. a table) on mongodb and create an index with a [Time to Live](https://docs.mongodb.com/manual/tutorial/expire-data/) parameter (TTL). Besides the operation benefits not having to maintain this collection (records are purged after TTL expires) we can bring a temporal dimension to our model in order to cope with users changing patterns. With a TTL of e.g. 1 week and a new model trained every day, we can track clusters over time and dynamically adapt our fraud strategy as new transactions are being observed whilst keeping track of historical records
# MAGIC 
# MAGIC ```
# MAGIC >> mongo
# MAGIC >> use fraud
# MAGIC >> db.tiles.createIndex( { "createdAt": 1 }, { expireAfterSeconds: 604800 } )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ```
# MAGIC import com.mongodb.spark._
# MAGIC 
# MAGIC tiles
# MAGIC   .withColumn("createdAt", current_timestamp())  
# MAGIC   .write
# MAGIC   .format("mongo")
# MAGIC   .mode("append")
# MAGIC   .option("database", "fraud")
# MAGIC   .option("collection", "tiles")
# MAGIC   .save()
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC An online operation process could be monitoring new card transactions in real time by simply searching for specific H3 tiles of a given user via a simple mongo db search query
# MAGIC 
# MAGIC ```
# MAGIC use fraud
# MAGIC db.tiles.find({"user": "7f103b53-25b4-483d-81f2-e646d22930b2", "tile": "8A2A1008916FFFF"})
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC In the visualization below, we show an example of how change in customers' transactional behaviours could be tracked over time (thanks to our TTL), where any observed location stays active for a period of X days and wherein anomalous transactions can be detected in real time.

# COMMAND ----------

# MAGIC %md
# MAGIC ![window](https://github.com/databricks-industry-solutions/geoscan-fraud/raw/main/images/geoscan_window.gif)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Closing thoughts
# MAGIC 
# MAGIC Card fraud transactions will never be addressed by a one-size-fits-all model but should always make use of various indicators coming from different controls as part of an overhatching fraud prevention strategy. Often, this combines [AI models with a rule based systems](https://databricks.com/blog/2021/01/19/combining-rules-based-and-ai-models-to-combat-financial-fraud.html), integrates advanced technologies and legacy processes, cloud based infrastructures and on premises systems, and must comply with tight regulatory requirements and critical SLAs. Although our approach does not aim at identifying fraudulent transactions on its own, it strongly contributes at extracting anomalous events in an **timely**, **cost effective** (self maintained) and fully **explainable** manner, hence a great candidate to combat financial fraud more effectively in a coordinated rules + AI strategy.
# MAGIC 
# MAGIC As part of this exercise, we also discovered something equally important in financial services. We demonstrated the ability of a Lakehouse infrastructure to transition from traditional to personalized banking where consumers are no longer segmented by demographics (who they are) but by their spending patterns (how they bank), paving the way towards a more customer centric future of retail banking.
