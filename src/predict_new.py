import pickle
import numpy as np
import skimage.io
from skimage.transform import resize
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image
import boto3
from mapping_functionality import gps_find, plot_map


def process_image(img):
    img = img.astype(np.float32) / 255.0
    img -= 0.5
    img * 2
    img_resized = resize(img, (299, 299, 3), mode='constant')
    img_batch = np.expand_dims(img_resized, axis =0)

    return img_batch

def predict_class(img):

    with open('../../tmp/CS3_model.pickle', 'rb') as fileobject:
        model = pickle.load(fileobject)

    model.load_weights("../../tmp/1544721035.51357.hdf5")

    probs = model.predict(img)
    top_prediction = np.argsort(probs)[0][::-1][0]

    classes = {0:'box elder beetle', 1:'cucumber beetle', 2:'emerald ash borer',
                3:'Japanese beetle',  4:'ladybug', 5:'striped cucumber beetle' }

    return classes[top_prediction]

def add_data_to_map(filepath):
    s3 = boto3.resource("s3")
    s3.Bucket('image-location-data').download_file('testfile', 'gps_pickle')
    with open('gps_pickle', 'rb') as fileobject:
        gps_df = pickle.load(fileobject)

    # add new data to df
    image = Image.open(filepath)
    if filepath[-3:] == 'jpg':
        info = image._getexif()
        coordinates = gps_find(info)
        if coordinates is not None:
            gps_df = gps_df.append(coordinates)
            print(coordinates)
    print(gps_df)
    new_map = plot_map(gps_df)
    plt.savefig('static/beetle_map')

    with open("gps_pickle", "wb") as fileobject:
        pickle.dump(gps_df, fileobject)

    s3 = boto3.client("s3")
    s3.upload_file('gps_pickle', 'image-location-data', 'testfile')
    s3.upload_file('beetle_map.png', 'image-location-data', 'beetle_map')

if __name__ == '__main__':
    filepath = "uploads/20181129_132810.jpg"

    image = skimage.io.imread(filepath)
    image = process_image(image)

    beetle_type = predict_class(image)

    if beetle_type == 'Japanese beetle':
        add_data_to_map(filepath)

    print(beetle_type)
