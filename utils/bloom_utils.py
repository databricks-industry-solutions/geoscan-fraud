import pybloomfilter
import mlflow


def bloom_version():
  return '.'.join([str(i) for i in list(pybloomfilter.VERSION)])

def train_bloom_filter(records):
    cluster = pybloomfilter.BloomFilter(len(records), 0.01)
    cluster.update(records)
    return cluster
  
class H3Lookup(mlflow.pyfunc.PythonModel):
  
    def __init__(self, user_df):
        self.user_df = user_df
    
    def load_context(self, context): 
        import pandas as pd
        import pybloomfilter    
        blooms = {}
        tiles = pd.read_csv(context.artifacts['tiles'])
        for i, rec in self.user_df.iterrows():
            records = list(rec.tiles)
            bloom = pybloomfilter.BloomFilter(len(records), 0.1)
            bloom.update(records)
            blooms[rec.user] = bloom
        self.blooms = blooms

    def predict(self, context, df):

        import h3
        def to_h3(x):
            h = h3.geo_to_h3(x[0], x[1], 10)
            return h.upper()

        def is_anomalous(x):
            if x[1] in self.blooms[x[0]]:
                return 0
            else:
                return 1

        df['h3'] = df[['latitude', 'longitude']].apply(to_h3, axis=1)
        df['anomaly'] = df[['user', 'h3']].apply(is_anomalous, axis=1)
        return df.drop(['h3'], axis=1)