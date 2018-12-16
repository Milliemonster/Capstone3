import numpy as np
np.random.seed(42)
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
#from image_process_cs2 import data_preprocess
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score
import matplotlib
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import datetime
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
ts = str(datetime.datetime.now().timestamp())
matplotlib.use('agg')

def create_model(nb_classes, img_rows, img_cols, img_layers):
    '''assembles CNN model layers
    inputs: number of classes, image size (rows, cols, layers)
    outputs: model
    '''

    nb_filters = 24
    pool_size = (2, 2)
    kernel_size = (4, 4)
    input_shape = (img_rows, img_cols, img_layers)

    model = Sequential()

    model.add(Conv2D(nb_filters, (4, 4), padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (4, 4), padding='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.3))

    model.add(Conv2D(nb_filters, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (1, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (5, 1), padding='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.3))

    model.add(Flatten())
    #shape = int(model.output_shape[1]*0.8)
    shape = 8000

    model.add(Dense(shape))
    model.add(Activation('relu'))

    model.add(Dropout(0.3))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

def generate_data(train_directory, validation_directory, test_directory, img_rows, img_cols, mode = 'rgb'):
    '''creates data generators for train, validation and test data
        inputs: paths to image data. Folders should be named with the target.
                image size (row, cols) and color mode
        outputs: three data generators.
        '''

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True
        )

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )

    train_generator = train_datagen.flow_from_directory(
        directory=train_directory,
        #save_to_dir = "japanese_beetle/train",
        target_size=(img_rows, img_cols),
        color_mode=mode,
        batch_size=16,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory=validation_directory,
        target_size=(img_rows, img_cols),
        color_mode=mode,
        batch_size=216,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_directory,
        target_size=(img_rows, img_cols),
        color_mode=mode,
        batch_size=125,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    return train_generator, test_generator, validation_generator

def make_analysis(generator):
    '''
    Computes accuracy of model and displays labeled images for incorrect guesses.
    inputs: test data generator
    outputs: balanced accuracy score
    '''

    test_X = generator[0][0]
    test_y = generator.classes

    predicted_y = model.predict_classes(test_X)
    probs = model.predict_proba(test_X).round(2)

    labels = np.vstack((test_y, predicted_y))
    results = np.hstack((probs, labels.T))

    classes = {0:'box elder beetle', 1:'spotted cucumber beetle', 2:'emerald ash borer',
                3:'Japanese beetle',  4:'ladybug', 5:'striped cucumber beetle' }

    score = balanced_accuracy_score(test_y, predicted_y)

    wrong_indices = []

    for i, prediction in enumerate(predicted_y):
        if prediction != test_y[i]:
            wrong_indices.append(i)

    for index in wrong_indices:
        plt.imshow((test_X[index]/2+0.5))
        plt.text(0.05, 0.95, f'Predicted: {classes[predicted_y[index]]} \n Actual: {classes[test_y[index]]}', fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.show()

    return score

def show_confusion(generator):
    test_X = generator[0][0]
    test_y = generator.classes
    predicted_y = model.predict_classes(test_X)

    class_names = ['box elder beetle', 'spotted cucumber beetle', 'emerald ash borer',
                'Japanese beetle',  'ladybug', 'striped cucumber beetle' ]

    cnf_matrix = confusion_matrix(test_y, predicted_y)
    np.set_printoptions(precision=2)
    print(cnf_matrix)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    plt.savefig('./'+ts+'confusion_matrix.png')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.savefig('./'+ts+'normalized_confusion_matrix.png')

if __name__ == '__main__':
    train_directory = "../../images/select/train"
    test_directory = "../../images/select/holdout"
    validation_directory = "../../images/select/validation"

    model = create_model(6, 100, 100, 3)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    ts = str(datetime.datetime.now().timestamp())
    checkpointer = ModelCheckpoint(filepath='../../tmp/'+ts+'.hdf5', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(
                log_dir='logs/', histogram_freq=0, batch_size=50, write_graph=True, embeddings_freq=0)

    train_generator, test_generator, validation_generator = generate_data(train_directory, validation_directory, test_directory, 100, 100)

    load = input("Load saved weights? (y/n) ")

    if load.lower() == 'y':
        model.load_weights("../../tmp/stable11.hdf5")
        print("weights loaded")
    elif load.lower() == 'n':
        model.fit_generator(train_generator,
                steps_per_epoch=200,
                epochs=15,
                validation_data=validation_generator,
                validation_steps=1, callbacks=[checkpointer, tensorboard])
        model.load_weights('../../tmp/'+ts+'.hdf5')

    score = make_analysis(validation_generator)
    print(f'balanced accuracy score is {score}')

    show_confusion(test_generator)
