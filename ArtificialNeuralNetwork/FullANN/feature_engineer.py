import pandas as pd
import numpy as np

class FeatureEngineer:
  def __init__(self, df):
    self.df = df

  def haversine_distance(self, df, coord_labels1, coord_labels2):
    earth_radius = 6371  # average radius of Earth in kilometers

    lat1, long1 = coord_labels1
    lat2, long2 = coord_labels2

    phi1 = np.radians(self.df[lat1])
    phi2 = np.radians(self.df[lat2])
    delta_phi = np.radians(self.df[lat2]-df[lat1])
    delta_lambda = np.radians(self.df[long2]-self.df[long1])

    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = (earth_radius * c)

    return distance

  def convert_coordinates_to_distance_travelled(self):
    pickup_coord_labels = ['pickup_latitude', 'pickup_longitude']
    dropoff_coord_labels = ['dropoff_latitude', 'dropoff_longitude']
    self.df['dist_km'] = self.haversine_distance(self.df, pickup_coord_labels, dropoff_coord_labels)

  def convert_datetime_to_useful_time_info(self):
    self.df['EDTdate'] = pd.to_datetime(self.df['pickup_datetime'].str[:19]) - pd.Timedelta(hours=4) 
    self.df['Hour'] = self.df['EDTdate'].dt.hour
    self.df['AMorPM'] = np.where(self.df['Hour']<12, 'am', 'pm')
    self.df['Weekday'] = self.df['EDTdate'].dt.strftime("%a")
