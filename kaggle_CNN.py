import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

# Read the input from the csv files
train = pd.read_csv('./data_set/train.csv')

# Get the labels from the training set
labels = train['label'].values
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)


def build_model():
    # Define constants
    batch_size = 128
    num_classes = 24
    epochs = 50

    unique_val = np.array(labels)
    np.unique(unique_val)

    # Remove first column in the training set
    train.drop('label', axis=1, inplace=True)

    images = train.values
    images = np.array([np.reshape(i, (28, 28)) for i in images])
    images = np.array([i.flatten() for i in images])

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)

    # Normalize the data
    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Build the CNN
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.20))
    model.add(Dense(num_classes, activation = 'softmax'))

    model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return loaded_model


def run_model(model, test):
    # test = pd.read_csv('./data_set/test.csv')
    # test_labels = test['label']
    # test.drop('label', axis=1, inplace=True)
    # test_images = test.values
    test_images = np.array([np.reshape(i, (28, 28)) for i in test])
    test_images = np.array([i.flatten() for i in test_images])
    test_images = test.reshape(test_images.shape[0], 28, 28, 1)
    y_pred = model.predict(test_images).round()[0]
    return_statement = np.where(y_pred == 1)[0][0]
    return return_statement
    # print(accuracy_score(test_labels, y_pred.round()))
