#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image
import boto3
import matplotlib as mpl


def gps_find(info):
    '''
    Pulls gps coordinates from Exif tags

    Parameters
    ----------
    info: dict

    Returns
    -------
    dataframe

    '''

    if info.get(34853, None) == None:
        return None
    else:
        positions = [info[34853][2], info[34853][4]]
        coordinates = []
        for pos in positions:
            gps_coord = pos[0][0]/pos[0][1] + pos[1][0]/(pos[1][1]*60) + pos[2][0]/(pos[2][1]*3600)
            coordinates.append(gps_coord)
        return pd.DataFrame([{'latitude':coordinates[0], 'longitude':-1*coordinates[1]}])

def plot_map(coordinates, new_coordinates):
    '''
    Plots a map of japanese beetle sightings in Colorado. Plots new sighting in
    blue and all others in red.

    Parameters
    ----------
    coordinates: dataframe
    new_coordinates: dataframe

    Returns
    -------
    m: basemap object
    '''

    mpl.rcParams.update({
        'font.size'           : 20.0,
        'axes.titlesize'      : 'x-large',
        'axes.labelsize'      : 'medium',
        'xtick.labelsize'     : 'medium',
        'ytick.labelsize'     : 'medium',
        'legend.fontsize'     : 'x-small',
        'legend.borderpad'    :  0.5,
        'legend.handletextpad':  0,
    })

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
    plt.title('Japanese beetle observations')
    old = m.plot(x, y, 'r.', label ='Existing sightings')

    if new_coordinates is not None:
        new_lats = new_coordinates.latitude.values
        new_lons = new_coordinates.longitude.values
        new_x, new_y = m(new_lons, new_lats)
        new = m.plot(new_x, new_y, 'b.', label = 'Your data')

    plt.legend()
    return(m)

if __name__ == '__main__':

    s3 = boto3.resource("s3")
    s3.Bucket('image-location-data').download_file('testfile', 'gps_pickle')
    with open("gps_pickle", 'rb') as fileobject:
        gps_df = pickle.load(fileobject)

    image = Image.open("/Users/millie/downloads/20181218_093137.jpg")
    info = image._getexif()
    coordinates = gps_find(info)
    if coordinates is not None:
        gps_df = gps_df.append(coordinates)
    print(gps_df)
    new_map = plot_map(gps_df, coordinates)
    plt.savefig('beetle_map')

    with open("gps_pickle", "wb") as fileobject:
        pickle.dump(gps_df, fileobject)

    s3 = boto3.client("s3")
    s3.upload_file('gps_pickle', 'image-location-data', 'testfile')
    s3.upload_file('beetle_map.png', 'image-location-data', 'beetle_map')
