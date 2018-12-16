# coding=utf-8
import os
import glob
import numpy as np
import matplotlib as mpl
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from predict_new import *
import tensorflow as tf

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
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
            beetle_type = predict_class(image)

            if beetle_type == 'Japanese beetle':
                add_data_to_map(path)
                return render_template('beetle.html', data=beetle_type)

            else:
                return render_template('other.html', data=beetle_type)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083, debug=False, threaded=True)
