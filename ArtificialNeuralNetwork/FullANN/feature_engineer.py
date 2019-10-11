import pandas as pd
import numpy as np
from haversine_formula import Haversine

class FeatureEngineer:
  def __init__(self, df):
    self.df = df

  def convert_coordinates_to_distance_travelled(self):
    pickup_coord_labels = ['pickup_latitude', 'pickup_longitude']
    dropoff_coord_labels = ['dropoff_latitude', 'dropoff_longitude']
    self.df['dist_km'] = Haversine.distance(self.df, pickup_coord_labels, dropoff_coord_labels)

  def convert_datetime_to_useful_time_info(self):
    self.df['EDTdate'] = pd.to_datetime(self.df['pickup_datetime'].str[:19]) - pd.Timedelta(hours=4) 
    self.df['Hour'] = self.df['EDTdate'].dt.hour
    self.df['AMorPM'] = np.where(self.df['Hour']<12, 'am', 'pm')
    self.df['Weekday'] = self.df['EDTdate'].dt.strftime("%a")