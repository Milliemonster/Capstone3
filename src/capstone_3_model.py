import numpy as np
np.random.seed(42)
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications import Xception
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from keras.applications.xception import preprocess_input
from keras.applications import Xception
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score
import matplotlib
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import datetime
import pickle
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
ts = str(datetime.datetime.now().timestamp())
#matplotlib.use('agg')

def create_transfer_model(input_size, n_categories, weights = 'imagenet'):
    '''
    Builds transfer learning model based on Xception https://arxiv.org/abs/1610.02357

    Parameters
    ----------
    input_size: tuple
    n_categories: int
    weights: hf5 file

    Returns
    -------
    model: Keras model
    '''
        base_model = Xception(weights=weights,
                          include_top=False,
                          input_shape=input_size)

        model = base_model.output
        model = GlobalAveragePooling2D()(model)
        predictions = Dense(n_categories, activation='softmax')(model)
        model = Model(inputs=base_model.input, outputs=predictions)

        return model

def change_trainable_layers(model, trainable_index):
    '''
    Changes the number of layers that are available for training
    beginning with trainable_index

    Parameters
    ----------
    model: Keras model
    trainable_index: int

    '''
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

def generate_data(train_directory, validation_directory, test_directory, img_rows, img_cols, mode = 'rgb'):
    '''
    Creates data generators for train, validation and test data
    Input strings are paths to image data. Folders should be named with the target.

    Parameters
    ----------
    train_directory: str
    validation_directory: str
    test_directory: str
    img_rows: int
    img_cols: int
    mode: str

    Returns
    -------
    train_generator, test_generator, validation_generator: Keras ImageDataGenerators
    '''

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.25,
        zoom_range=0.25,
        width_shift_range = 0.25,
        height_shift_range = 0.25,
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
        batch_size=228,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_directory,
        target_size=(img_rows, img_cols),
        color_mode=mode,
        batch_size=113,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    return train_generator, test_generator, validation_generator

def make_analysis(generator):
    '''
    Computes accuracy of model and displays labeled images for incorrect guesses.

    Parameters
    ----------
    generator: Keras ImageDataGenerator

    Returns
    -------
    score: int
    '''

    test_X = generator[0][0]
    test_y = generator.classes

    probs = model.predict(test_X)
    indices = probs.argsort(axis = 1)
    top_prediction = np.flip(indices, 1)[:, 0]
    top_prediction.reshape(1, -1)

    classes = {0:'box elder beetle', 1:'cucumber beetle', 2:'emerald ash borer',
                3:'Japanese beetle',  4:'ladybug', 5:'striped cucumber beetle' }

    score = balanced_accuracy_score(test_y, top_prediction)

    wrong_indices = []

    for i, prediction in enumerate(top_prediction):
        if prediction != test_y[i]:
            wrong_indices.append(i)

    for index in wrong_indices:
        plt.imshow((test_X[index]/2+0.5))
        plt.text(0.05, 0.95, f'Predicted: {classes[top_prediction[index]]} \n Actual: {classes[test_y[index]]}', fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.savefig('./result_images/'+ts+str(index)+'miss.png')
        plt.clf()

    return score

def show_confusion(generator):
    '''
    Plots confusion matrix for model predictions

    Parameters
    ----------
    generator: Keras ImageDataGenerator

    '''
    test_X = generator[0][0]
    test_y = generator.classes

    probs = model.predict(test_X)
    indices = probs.argsort(axis = 1)
    top_prediction = np.flip(indices, 1)[:, 0]
    top_prediction.reshape(1, -1)

    class_names = ['box elder beetle', 'cucumber beetle','emerald ash borer',
                'Japanese beetle', 'ladybug', 'striped cucumber beetle']

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_y, top_prediction)
    np.set_printoptions(precision=2)

    print(cnf_matrix)
    #Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    plt.savefig('./result_images/'+ts+'confusion_matrix.png')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.savefig('./result_images/'+ts+'normalized_confusion_matrix.png')

def fit_model(model):
    '''
    Fits the model to the training data. Uses weighted classes to address class imbalance

    Parameters
    ----------
    model: Keras model

    Returns
    -------
    model: fitted Keras model
    '''

    _ = change_trainable_layers(model, 132)
    model.compile(optimizer=RMSprop(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

    class_weights = {0: 0.4, 1: 0.47, 2: 0.69, 3: 0.41, 4:0.45, 5: 1}
    model.fit_generator(train_generator,
            steps_per_epoch=200,
            epochs=5,
            validation_data=validation_generator,class_weight = class_weights,
            validation_steps=1, callbacks=[checkpointer, tensorboard])
    model.load_weights('../../tmp/'+ts+'.hdf5')

    _ = change_trainable_layers(model, 126)
    model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_generator,
            steps_per_epoch=200,
            epochs=5,
            validation_data=validation_generator,
            validation_steps=1, callbacks=[checkpointer, tensorboard])
    model.load_weights('../../tmp/'+ts+'.hdf5')

    _ = change_trainable_layers(model, 116)
    model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_generator,
            steps_per_epoch=200,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=1, callbacks=[checkpointer, tensorboard])
    model.load_weights('../../tmp/'+ts+'.hdf5')

    return model

if __name__ == '__main__':
    train_directory = "../../images/select/train"
    test_directory = "../../images/select/holdout"
    validation_directory = "../../images/select/validation"

    model = create_transfer_model((299,299,3),6)

    checkpointer = ModelCheckpoint(filepath='../../tmp/'+ts+'.hdf5', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(
                log_dir='logs/', histogram_freq=0, batch_size=50, write_graph=True, embeddings_freq=0)

    train_generator, test_generator, validation_generator = generate_data(train_directory, validation_directory, test_directory, 299, 299)

    load = input("Load saved weights? (y/n) ")

    if load.lower() == 'y':
        model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights("static/1544982049.165118.hdf5")
        model.save("static/CS3_model.hdf5")
        print("weights loaded")

    elif load.lower() == 'n':
        model = fit_model(model)
        model.load_weights('../../tmp/'+ts+'.hdf5')

    score = make_analysis(validation_generator)
    print(f'balanced accuracy score is {score}')

    show_confusion(validation_generator)

    with open('../../tmp/CS3_model.pickle', 'wb') as fileobject:
        pickle.dump(model, fileobject)
