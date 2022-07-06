import h3
from pyspark.sql.functions import udf
from pyspark.sql import functions as F


@udf("string")
def to_h3(lat, lng, precision):
    h = h3.geo_to_h3(lat, lng, precision)
    return h.upper()