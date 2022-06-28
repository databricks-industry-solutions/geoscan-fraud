# Databricks notebook source
# MAGIC %md
# MAGIC # Density based spatial clustering
# MAGIC [DBSCAN](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) (density-based spatial clustering of applications with noise) 
# MAGIC is a common clustering technique used to group points that are closely packed together. Compared to other clustering methodologies, 
# MAGIC it doesn't require you to indicate the number of clusters beforehand, can detect clusters of varying shapes and sizes 
# MAGIC and is strong at finding outliers that don't belong to any dense area, hence a great candidate for geospatial analysis of credit card 
# MAGIC transactions and fraud detection. This, however, comes with a serious price tag: DBSCAN requires all points to be compared 
# MAGIC to every other points in order to find dense neighborhoods where at least `minPts` points can be found within a `epsilon` radius. Here comes **GEOSCAN**, our novel approach to geospatial clustering.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Introducing GEOSCAN
# MAGIC 
# MAGIC As we could not find any viable solution that could scale to the millions of customers or to more than a few hundreds of thousands of records, we created our own open source library, [GEOSCAN](https://github.com/databrickslabs/geoscan), our implementation of DBSCAN algorithm for geospatial clustering at big data scale. Leveraging uber [H3](https://eng.uber.com/h3/) library to only group points we know are in close vicinity (according to H3 `precision`) and relying on [GraphX](https://spark.apache.org/docs/latest/graphx-programming-guide.html) API, this framework can detect dense areas at massive scale, understanding user shopping behaviours and detecting anomalous transactions in near real time. We will be using `folium` library to visualize our approach step by step as we move from a one size fits all model to a personalized clustering and anomaly detection. 
# MAGIC 
# MAGIC **Step1: Grouping**
# MAGIC 
# MAGIC The first step of GEOSCAN is to link each point to all its neighbours within an `epsilon` distance and remove points having less than `minPts` neighbours. Concretely, this means running a cartesian product - `O(n^2)` time complexity - of our dataset to filter out tuples that are more than `epsilon` meters away from one another. In our approach, we leverage H3 hexagons to only group points we know are close enough to be worth comparing. As reported in below picture, we first map a point to an H3 polygon and draw a circle of radius `epsilon` that we tile to at least 1 complete ring. Therefore, 2 points being at a distance of `epsilon` away would be sharing at least 1 polygon in common, so grouping by polygon would group points in close vicinity, ignoring 99.99% of the dataset. These pairs can then be further measured using a [haversine](https://en.wikipedia.org/wiki/Haversine_formula) distance.
# MAGIC 
# MAGIC <img src="https://github.com/databricks-industry-solutions/geoscan-fraud/raw/main/images/geoscan_algorithm.png" width=800>
# MAGIC 
# MAGIC Even though the theoretical time complexity remains the same - `O(n^2)` - we did not have to run an expensive (and non realistic) cartesian product of our entire dataframe. The real time complexity is `O(p.k^2)` where `p` groups are processed in parallel, running cartesian product of `k` points (`k << n`) sharing a same H3 hexagon, hence scaling massively. This isn't magic though, and prone to failure when data is heavily skewed in dense area (we recommend sampling data in specific areas as reported later). 
# MAGIC  
# MAGIC **Step2: Clustering**
# MAGIC 
# MAGIC The second step is trivial when using a graph paradigm. As we found vertices being no more than `epsilon` meters away (edge contains distance), we simply remove vertices with less than `minPts` connections (`degrees < minPts`). By removing these border nodes, clusters start to form and can be retrieved via a `connectedComponents`. 
# MAGIC 
# MAGIC **Step3: Convex Hulls**
# MAGIC 
# MAGIC As all our core points are defining our clusters, the final step is to find the [Convex Hull](https://en.wikipedia.org/wiki/Convex_hull), that is the smallest shape that include all of our core geo coordinates. There are plenty of litterature on that topic, and our approach can easily be used in memory for each cluster returned by our connected components. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dependencies

# COMMAND ----------

# MAGIC %md
# MAGIC In addition to GEOSCAN jar file that must be installed on classpath, we also need to install its python wrapper. Installed in the future via a pypi repo, one needs to install local files from git whilst our code is not yet published.

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploring data
# MAGIC As consumers are becoming more and more digitally engaged, large financial service institutions often have access to GPS coordinates of every purchase made by their customers, in real time. With around 40 billions of card transactions being processed in the US every year, retail banks have a lot of data they could leverage to better understand customers's behaviors over time (for customers opting in GPS enabled apps). However, it often requires access to large amount of resources and cutting edge libraries to run expensive geospatial computations that do not "fit" well with a traditional data warehouse paradigm. In this example, we generated half a million of synthetic data points of geo coordinates for multiple users. 

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

# MAGIC %md 
# MAGIC Before running our geospatial clustering, it may be worth understanding our data better. We enrich our data with H3 polygons of different dimensions to identify potential skews in our distribution, these high street places or large retail mall "attracting" most of transactions,  acting as evident bottlenecks for large `epsilon` values. In addition to the `GeoUtils` class available on GEOSCAN package, H3 can also be used natively via a python API.

# COMMAND ----------

import h3
from pyspark.sql.functions import udf
from pyspark.sql import functions as F

@udf("string")
def to_h3(lat, lng, precision):
  h = h3.geo_to_h3(lat, lng, precision)
  return h.upper()

display(
  points_df
    .groupBy(to_h3(F.col('latitude'), F.col('longitude'), F.lit(9)).alias('h3'))
    .count()
    .orderBy(F.desc('count'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC At resolution 9 (the resolution table can be found [here](https://h3geo.org/docs/core-library/restable)), or approximately 150m radius, we observe a few areas with about 3,000 observations. At such a granularity, our model would still need to compute 3,000 x 3,000 pairwise distances. Although this is by far better than 500,000 x 500,000 that would be required with a traditional DBSCAN approach, we show later how to sample our dataset geographically to remove possible skews whilst maintaining GEOSCAN predictive power. 

# COMMAND ----------

import folium
from folium import plugins

points = points_df.sample(0.1).toPandas()[['latitude', 'longitude']]
nyc = folium.Map([40.75466940037548,-73.98365020751953], zoom_start=12, width='80%', height='100%')
folium.TileLayer('Stamen Toner').add_to(nyc)
nyc.add_child(plugins.HeatMap(points.to_numpy(), radius=12))
nyc

# COMMAND ----------

# MAGIC %md
# MAGIC Our synthetic data set exhibits denser areas around Chelsea, East village and the financial district. By zooming in, we can reveal well defined zones that we aim at programmatically extracting using GEOSCAN

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed clustering
# MAGIC Working **fully distributed**, we retrieve clusters from an entire dataframe (i.e. across all our users) using the Spark ML API, hence fully compliant with the Spark Pipeline framework (model can be serialized / deserialized). In this mode, the core of GEOSCAN algorithm relies on GraphX to detect points having `distance < epsilon` and a `degree > minPoints`. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model training
# MAGIC 
# MAGIC We will be using a relatively small `epsilon` value at first to overcome the skews observed earlier. Furthermore, given the amount of data we have in dense areas, having `minPts` too low would result in the entire shape of NYC to be returned as one cluster. How do we tune `epsilon`? Largely domain-specific and with no established strategy, a rule of thumb could be to plot k nearest neighbors, looking at distances and choosing the point of max curvature (more [information](https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc)). We leave this to the discretion of the reader. We will try different approaches with different values of `epsilon` and `minPts`, using folium to visualize and refine our clustering strategy

# COMMAND ----------

from geoscan import Geoscan
import mlflow

with mlflow.start_run(run_name='GEOSCAN') as run:

  geoscan = Geoscan() \
    .setLatitudeCol('latitude') \
    .setLongitudeCol('longitude') \
    .setPredictionCol('cluster') \
    .setEpsilon(200) \
    .setMinPts(20)
  
  mlflow.log_param('epsilon', 200)
  mlflow.log_param('minPts', 20)
  
  model = geoscan.fit(points_df)
  mlflow.spark.log_model(model, 'model')
  run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC As strong advocate of open standard, we build GEOSCAN to support GeoJSON [rfc7946](https://tools.ietf.org/html/rfc7946) as model output. For convenience, we can attach GeoJson file as an artifact alongside the model on mlflow (file will be visualized as-is on mlflow tracking server).

# COMMAND ----------

geoJson = model.toGeoJson()
with open("/tmp/{}_geoscan.geojson".format(run_id), 'w') as f:
  f.write(geoJson)

import mlflow
client = mlflow.tracking.MlflowClient()
client.log_artifact(run_id, "/tmp/{}_geoscan.geojson".format(run_id))

# COMMAND ----------

# MAGIC %md
# MAGIC With our model exported as GeoJson object, we can overlay our clusters on a same `folium` visualization.

# COMMAND ----------

folium.GeoJson(geoJson).add_to(nyc)
nyc

# COMMAND ----------

# MAGIC %md
# MAGIC One can play with different `epsilon` and `minPts` values resulting in clusters of different sizes and shapes. As discussed, tuning geospatial clustering model mainly relies on domain expertise than golden standard rule. As represented above, our parameters resulted in a relative large shape covering most of Manhattan. Although reducing `minPts` value could help refining that shape, it may certainly impact less dense areas. In addition to performance gain, sampling our data may become handy if not necessary.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance tuning
# MAGIC Given the skews observed in our data distribution, it is expected to take more time for algoritm to group points to their nearest neighborood with large `epsilon` values. Although we clearly beat the `O(N^2)` curse of DBSCAN with well distributed data, training on skewed dataset tend to same time complexity (minus the technical limits imposed by memory) as `n` points would share same polygons `P x O(n^2) = O(n^2)`. Using simple UDF and native H3 library, one could reduce the complexity by sampling transactions to maximum of X points within a same radius (we will be using a sampling resolution of 11)

# COMMAND ----------

import random
from pyspark.sql.types import *

# we randomly select maximum 10 points within a same polygon of size 11 (30m)
def sample(latitudes, longitudes):
  l = list(zip(latitudes, longitudes))
  return random.sample(l, min(len(l), 10))

sample_schema = ArrayType(StructType([StructField("latitude", DoubleType()), StructField("longitude", DoubleType())]))
sample_udf = udf(sample, sample_schema)

sample_df = (
  points_df
    .groupBy(to_h3(F.col("latitude"), F.col("longitude"), F.lit(11)))
    .agg(F.collect_list(F.col("latitude")).alias("latitudes"), F.collect_list(F.col("longitude")).alias("longitudes"))
    .withColumn('sample', F.explode(sample_udf(F.col('latitudes'), F.col('longitudes'))))
    .select('sample.latitude', 'sample.longitude')
)

display(
  sample_df
    .groupBy(to_h3(F.col("latitude"), F.col("longitude"), F.lit(9)).alias("h3"))
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC By taking at most 10 random point in each polygon of size 30m, we drastically dropped our skew by 80%, resulting in a much more even distribution of our data. Still, with at most 10 points per 30m, we still satisfy our GEOSCAN criteria (`10 > minPts` and `30 < epsilon`). This, of course, is a simple example and would require further understanding on the data distribution and a possible dynamic sampling strategy for different areas.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model inference
# MAGIC 
# MAGIC As the core of GEOSCAN logic relies on the use of H3 polygons, it becomes natural to leverage the same for model inference instead of bringing in extra GIS dependencies for expensive [point in polygons](https://en.wikipedia.org/wiki/Point_in_polygon) queries. Our approach consists in "tiling" our clusters with H3 hexagons that can easily be joined to our original dataframe. The logic is abstracted through the `transform` method of our `Estimator` Spark interface.

# COMMAND ----------

from pyspark.sql import functions as F

display(
  model
    .transform(points_df)
    .groupBy('cluster')
    .count()
    .orderBy(F.asc('cluster'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC We do not seem to get much more insights using a one size-fits-all clustering strategy across our entire customer base as most of transactions happens in NYC central. However, we could wonder where could we find our 2,000 "non clustered" transactions. Could we consider those as possible anomalous transactions?

# COMMAND ----------

from folium.plugins import MarkerCluster

nyc_anomalies_points = model.transform(points_df).filter(F.expr('cluster IS NULL')).sample(0.01).toPandas()
nyc_anomalies = folium.Map([40.75466940037548,-73.98365020751953], zoom_start=12, width='80%', height='100%')
folium.TileLayer('Stamen Toner').add_to(nyc_anomalies)
folium.GeoJson(geoJson, name="geojson").add_to(nyc_anomalies)
for _, point in nyc_anomalies_points.iterrows():
  folium.CircleMarker([point.latitude, point.longitude], radius=2, color='red').add_to(nyc_anomalies)

nyc_anomalies

# COMMAND ----------

# MAGIC %md
# MAGIC Given that clusters are density based, it is expected to find un-clustered points located near the edges of our clusters, probably still `epsilon` meters away from their neighbours but having less than `minPts` neighbours. In order to accomodate fraud detection use cases, we may want to expand our clusters slightly to incorporate transactions at a close vicinity.

# COMMAND ----------

# MAGIC %md
# MAGIC Supporting the Spark ML API, our model can be serialized / deserialized as-is, outputing data as a GeoJson file as previously discussed.

# COMMAND ----------

dbutils.fs.rm("/tmp/{}_geoscan".format(run_id), True)
model.save("/tmp/{}_geoscan".format(run_id))

# COMMAND ----------

from geoscan import GeoscanModel
model = GeoscanModel.load("/tmp/{}_geoscan".format(run_id))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Personalized clustering
# MAGIC In the previous section, we demonstrate how GEOSCAN can be used across our entire dataset. However, the aim was not to machine learn the NYC shape file, nor to find the best place to go shopping, but to track user shopping behaviour over time, where they may live, work, shop, etc. and where transactions are the least expected to happen in order to identify anomalous events. GEOSCAN supports a **pseudo distributed** approach where millions of models can be trained in parallel for millions of customers. Given that we drastically reduce the number of data to be processed for each user, we can afford to be much more targeted with higher `epsilon` values and lower `minPts`

# COMMAND ----------

from geoscan import GeoscanPersonalized
import mlflow

with mlflow.start_run(run_name='GEOSCAN_PERSONALIZED') as run:

  geoscan = GeoscanPersonalized() \
    .setLatitudeCol('latitude') \
    .setLongitudeCol('longitude') \
    .setPredictionCol('cluster') \
    .setGroupedCol('user') \
    .setEpsilon(100) \
    .setMinPts(3)

  models = geoscan.fit(points_df)
  
  mlflow.log_param('epsilon', 100)
  mlflow.log_param('minPts', 3)
  run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC Training 200 models in parallel tooks only a couple of minutes on our entire dataset and on a small policy cluster. Note that our Spark model is no longer returning a unique model but a collection of GeoJson objects that can be accessed via a spark dataframe and stored on Delta table. Similar to our distributed approach, models can be stored and retrieved as per standard Spark API as follows. One caveat is that - instead of returning an in memory object - our model returns a dataframe that will be re-evaluated to subsequent actions. We therefore recomment persisting it / reloading first.

# COMMAND ----------

model_path = config['model']['path']

# COMMAND ----------

dbutils.fs.rm(model_path, True)
models.save(model_path)

# COMMAND ----------

from geoscan import GeoscanPersonalizedModel
model_personalized = GeoscanPersonalizedModel.load(model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of one large GeoJson object, we access a dataframe object for each user and its respective clusters as GeoJson formatted records

# COMMAND ----------

geoJsons = model_personalized.toGeoJson()
display(geoJsons)

# COMMAND ----------

# MAGIC %md
# MAGIC With all records available, one can easily select a specific slice for a given user and overlay with its respecting transactions as a heatmap

# COMMAND ----------

from pyspark.sql import functions as F

user = '9cafdb6d-9134-4ee8-bdf6-972ebc3af729'
personalized_geojson = geoJsons.filter(F.col('user') == user).toPandas().iloc[0].cluster
personalized_data = points_df.filter(F.col('user') == user).toPandas()[['latitude', 'longitude']]

nyc_personalized = folium.Map([40.75466940037548,-73.98365020751953], zoom_start=12, width='80%', height='100%')
folium.TileLayer('Stamen Toner').add_to(nyc_personalized)
nyc_personalized.add_child(plugins.HeatMap(personalized_data.to_numpy(), radius=8))
folium.GeoJson(personalized_geojson, name="geojson").add_to(nyc_personalized)
nyc_personalized

# COMMAND ----------

# MAGIC %md
# MAGIC As reported in the above picture, we've gained further insights around a specific user's behaviour. This indicates 4 zones where this user may be the most likely to shop. Although this visualization flags what anyone could easily eye-ball, doing so programmatically was not an easy undertaking and helps us leverage the `Estimator` interface to classify all transactions for every users to their respective clusters in an easy and performant way.

# COMMAND ----------

display(model_personalized.transform(points_df))

# COMMAND ----------

# MAGIC %md
# MAGIC Although a transaction happening outside of any of these clusters could not necessarily be fraudulent, such anomalous patterns would be worth flagging as a potential feature that can be combined in an overhatching [framework](https://databricks.com/blog/2021/01/19/combining-rules-based-and-ai-models-to-combat-financial-fraud.html) to combat financial fraud. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding customers patterns
# MAGIC Before investigating our fraud use case further, it is important to step back and reflect on the insights we have been able to gain so far. As we have been able to learn more about our entire customer base (distributed approach), we could leverage this information to better understand the behaviour that are specific to each individual. If everyone were to shop at a same location, such an area would be less specific to a particular user. We can detect "personalized" zones as how much they overlap with common areas, better understanding our customers and paving the way towards truly personalized banking.

# COMMAND ----------

# MAGIC %md
# MAGIC We extended our Spark models to support a `getTiles` method that (as it says on the tin) will "fill" our polygons with H3 tiles of a given precision. Furthermore, taking into consideration our edge conditions, one can relax our boundary by allowing tiles to slightly spill over by 1,2, or X additional layers, capturing nearby transactions. Note that high H3 resolution will better fit a polygon but also requires more memory to keep all tiles, so one may have to balance between accuracy and memory constraints. Whilst you can store tiles as-is and move to next notebook, we want to explore that concept of personalized finance a bit more.

# COMMAND ----------

personalized_tiles = model_personalized.getTiles(precision=10, layers=5)
display(personalized_tiles)

# COMMAND ----------

# MAGIC %md
# MAGIC We represent below our tiling methodology with additional 2 layers, stretching our Geoshape (solid line) by a couple of hundreds meters to capture nearby transactions

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/geoscan/geoscan_tiling.png' width=500>

# COMMAND ----------

# MAGIC %md
# MAGIC Detecting areas that are the most descriptive for each user is similar to detecting keywords that are more descriptive to each sentence in Natural Language processing use cases. We can use a Term Frequency / Inverse document frequency ([TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)) approach to increase weight of user specific locations whilst reducing weight around common areas. We retrieve the number of unique visitor per H3 tile (`df`) and the total number of visits for each user in those same tiles (`tf`). 
# MAGIC 
# MAGIC $${tfidf}_{i,j} = {tf}_{i,j}\cdot log(\frac{N}{{df}_{i}})$$

# COMMAND ----------

points_h3 = points_df.select(F.col('user'), to_h3(F.col('latitude'), F.col('longitude'), F.lit(10)).alias('h3'))
document_frequency = (
  personalized_tiles
    .drop('user')
    .join(points_h3, ['h3'])
    .select('user', 'h3')
    .distinct()
    .groupBy('h3')
    .agg(F.sum(F.lit(1)).alias('df'))
)

# COMMAND ----------

term_frequency = (
  personalized_tiles
    .join(points_h3, ['h3', 'user'])
    .groupBy('user', 'h3', 'cluster')
    .agg(F.sum(F.lit(1)).alias('tf'))
)

# COMMAND ----------

import math
n = sc.broadcast(document_frequency.count())

@udf('double')
def tf_idf(tf, df):
  return tf * math.log(n.value / df)

personalized_areas = (
  term_frequency
    .join(document_frequency, ['h3'])
    .withColumn('tf_idf', tf_idf(F.col('tf'), F.col('df')))
    .select('user', 'cluster', 'h3', 'tf_idf')
)

display(personalized_areas)

# COMMAND ----------

# MAGIC %md
# MAGIC By storing all our clusters tiled with H3 polygons, we created a single reference lookup table for any known behavioural pattern. Given a specific user and a H3 location of size 10, we could detect if this location is part of any known and descriptive pattern for that user or if this activity is worth flagging as a unusual behavior. Furthermore, it is worth mentioning that our tiles being stored on Delta lake, it becomes easier to understand previous behaviour and trends using time travel functionality

# COMMAND ----------

personalized_areas.write.format('delta').mode('overwrite').saveAsTable(config['database']['tables']['tiles'])

# COMMAND ----------

# MAGIC %md
# MAGIC For faster lookup, we optimize our table for user and H3 access. 

# COMMAND ----------

sql("OPTIMIZE {} ZORDER BY (user, h3)".format(config['database']['tables']['tiles']))

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we can represent the same shapes as earlier for our user, but this time color coded by popularity. Darker is the color, the more descriptive this region is to the specified user.

# COMMAND ----------

personalized_tiles = spark.read.table(config['database']['tables']['tiles']).filter(F.col('user') == user)
display(personalized_tiles)

# COMMAND ----------

personalized_density = personalized_tiles.groupBy('cluster').agg(F.max('tf_idf').alias('max_tf_idf')).toPandas()[['cluster', 'max_tf_idf']]
personalized_geojson = geoJsons.filter(F.col('user') == user).toPandas().cluster.iloc[0]
data_bins = list(personalized_density.max_tf_idf.quantile([0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
nyc_personalized = folium.Map([40.75466940037548,-73.98365020751953], zoom_start=12, width='80%', height='100%')
folium.TileLayer('Stamen Toner').add_to(nyc_personalized)

# Color least popular areas by quantile
folium.Choropleth(
    geo_data = personalized_geojson,
    name='choropleth',
    data = personalized_density,
    columns=['cluster','max_tf_idf'],
    key_on='feature.id',
    fill_color='BuPu',
    fill_opacity=0.9,
    line_opacity=0.7,
    bins = data_bins
).add_to(nyc_personalized)

nyc_personalized

# COMMAND ----------

# MAGIC %md
# MAGIC We suddenly have gained incredible insights about our customer's shopping behaviour. Although the core of their transactions are made in the chelsea and the financial district area, what seems to better define this user are his / her transactions around the Plaza Hotel and Williamsburg. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paving the way to fraud detection
# MAGIC In this notebook, we have introduced a novel approach geospatial clustering in order to gain further insights on user shopping behaviour. We showed how to leverage the information from our entire customer base to better understand users' specific behaviours from large regions down to a few meters and demonstrated the importance to track customer changes over time using Delta as our underlying customer 360 strategy. Although we used a synthetic dataset, we showed that geospatial analysis can tell us a lot of information about customers behaviour, hence a critical component to anomaly detection and fraud prevention. In the next notebook, we will demonstrate how to use our "tiling" approach to detect suspicious behaviour in real time outside of a spark environment with high SLA and low latency requirements
