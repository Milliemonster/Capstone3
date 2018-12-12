from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image
import boto3


def gps_find(info):


    if info.get(34853, None) == None:
        return None
    else:
        positions = [info[34853][2], info[34853][4]]
        coordinates = []
        for pos in positions:
            gps_coord = pos[0][0]/pos[0][1] + pos[1][0]/(pos[1][1]*60) + pos[2][0]/(pos[2][1]*3600)
            coordinates.append(gps_coord)
        return pd.DataFrame([{'latitude':coordinates[0], 'longitude':-1*coordinates[1]}])

def plot_map(coordinates):
    lats = coordinates.latitude.values
    lons = coordinates.longitude.values

    fig = plt.gcf()
    fig.set_size_inches(8, 6.5)

    m = Basemap(projection='merc', \
                llcrnrlon=-110,llcrnrlat=36.5,urcrnrlon=-101.,urcrnrlat= 41.5, \
                resolution='i')

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates(color='0.2')
    m.fillcontinents(color = '#84b745', lake_color='#99ffff')
    m.drawmapboundary()

    x, y = m(lons, lats)
    m.plot(x, y, 'r.')

    return(m)

if __name__ == '__main__':

    s3 = boto3.resource("s3")
    s3.Bucket('image-location-data').download_file('testfile', 'gps_pickle')
    gps_df = pickle.load(open("gps_pickle", "rb" ) )
    #gps_pickle.close()

    # add new data to df
    image = Image.open("/Users/millie/downloads/DSC09521.jpg")
    info = image._getexif()
    coordinates = gps_find(info)
    if coordinates != None:
        gps_df = gps_df.append(coordinates)
    print(gps_df)
    new_map = plot_map(gps_df)
    plt.savefig('beetle_map')

    pickle.dump(gps_df, open("gps_pickle", "wb" ))
    #gps_pickle.close()

    s3 = boto3.client("s3")
    s3.upload_file('gps_pickle', 'image-location-data', 'testfile')
    s3.upload_file('beetle_map.png', 'image-location-data', 'beetle_map')
