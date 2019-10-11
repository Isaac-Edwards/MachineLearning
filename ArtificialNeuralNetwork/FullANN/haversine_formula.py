import numpy as np

class Haversine:
  ''' Haversine Formula - en.wikipedia.org/wiki/Haversine_formula'''
  def __init__(self):
    pass

  @staticmethod
  def distance(df, coord_labels1, coord_labels2):
    earth_radius = 6371  # average radius of Earth in kilometers

    lat1, long1 = coord_labels1
    lat2, long2 = coord_labels2

    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])

    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = (earth_radius * c)

    return distance