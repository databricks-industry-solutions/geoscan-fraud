# Databricks notebook source
# MAGIC %pip install geoscan==0.2.8 h3==3.7.2 folium==0.12.1 pybloomfiltermmap3==0.5.5

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import re
from pathlib import Path

# We ensure that all objects created in that notebooks will be registered in a user specific database. 
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username = useremail.split('@')[0]

# Please replace this cell should you want to store data somewhere else.
database_name = '{}_geoscan'.format(re.sub('\W', '_', username))
_ = sql("CREATE DATABASE IF NOT EXISTS {}".format(database_name))

# Similar to database, we will store actual content on a given path
home_directory = '/FileStore/{}/geoscan'.format(username)
dbutils.fs.mkdirs(home_directory)

# Where we might stored temporary data on local disk
temp_directory = "/tmp/{}/geoscan".format(username)
Path(temp_directory).mkdir(parents=True, exist_ok=True)

# COMMAND ----------

import re

config = {
  'db_raw_data': f'{database_name}.raw_transactions',
  'db_personalized_tiles': f'{database_name}.tiles',
  'model_name': database_name
}

# COMMAND ----------

import mlflow
experiment_name = f"/Users/{useremail}/geoscan_experiment"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

def tear_down():
  import shutil
  try:
    shutil.rmtree(temp_directory)
  except:
    pass
  dbutils.fs.rm(home_directory, True)
  _ = sql("DROP DATABASE IF EXISTS {} CASCADE".format(database_name))
