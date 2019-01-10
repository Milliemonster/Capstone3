import os
import glob
import numpy as np
import matplotlib as mpl
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from predict_new import *
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

model = load_model('static/CS3_model.hdf5')
model._make_predict_function()
UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    '''
    incorporates file upload code from: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
    '''
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            image = skimage.io.imread(path)
            image = process_image(image)

            beetle_type = predict_class(image, model)

            if beetle_type == 'Japanese beetle':
                _ = add_data_to_map(path)
                return render_template('beetle.html', data=beetle_type)

            else:
                classes = {'box elder beetle':0, 'cucumber beetle': 1, 'emerald ash borer': 2,
                            'ladybug':3, 'striped cucumber beetle':4}
                image_no = classes[beetle_type]
                return render_template('other.html', data=beetle_type, image = image_no )

    return render_template('index.html')

@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083, debug=False, threaded=True)
