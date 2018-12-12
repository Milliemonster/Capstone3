import flickrapi
import urllib
from PIL import Image
import pdb

# Flickr api access key
flickr=flickrapi.FlickrAPI('de16a8a26ce0053e805d7758a437b3fe', '964809ab7efbd4a5', cache=True)

keyword = 'green fly'

photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     tags=keyword,
                     extras='url_c',
                     per_page=500,           # may be you can try different numbers..
                     sort='relevance')

urls = []
for i, photo in enumerate(photos):
    url = photo.get('url_c')
    if url != None:
        urls.append(url)
    if i > 500:
        break

print (urls)

# Download image from the url and save it to '00001.jpg'
for i in range(len(urls)):
        print(urls[i])
        urllib.request.urlretrieve(urls[i], str(i)+".jpg")

# # Resize the image and overwrite it
# image = Image.open('00001.jpg')
# image = image.resize((256, 256), Image.ANTIALIAS)
# image.save('00001.jpg')
