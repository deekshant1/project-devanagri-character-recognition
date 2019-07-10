# import numpy as np
# from keras import Sequential
from keras.callbacks import ModelCheckpoint
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import print_summary
# import pandas as pd

# from pandas import DataFrame
# from pandas.io.parsers import TextFileReader
# from keras.utils.np_utils import to_categorical
import random
import numpy as np
import pandas as pd
# Visualisation imports
#import matplotlib.pyplot as plt
#import seaborn as sns
# Scikit learn for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Keras Imports - CNN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

data = pd.read_csv(r"C:\Users\hp\Downloads\devanagari-character-set\data.csv")
# dataset = np.array(data)
# np.random.shuffle(dataset)
# X = dataset
# Y = dataset
# X = X[:, 0:1024]
# Y = Y[:, 1024]


X = data.values[:, :-1] / 255.0
Y = data["character"].values

n_classes = 46

# Let's split the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# Encode the categories
le = LabelEncoder()
Y_train = le.fit_transform(y_train)
Y_test = le.transform(y_test)
train_y = to_categorical(Y_train, n_classes)
test_y = to_categorical(Y_test, n_classes)

# X_train = X[0:20000, :]
# X_train = X_train / 255.
# X_test = X[20000:22000, :]
# X_test = X_test / 255.

# # Reshape
# Y = Y.reshape(Y.shape[0], 1)
# Y_train = Y[0:20000, :]
# Y_train = Y_train.T
# Y_test = Y[20000:22000, :]
# Y_test = Y_test.T

# print("number of training examples = " + str(X_train.shape[0]))
# print("number of test examples = " + str(X_test.shape[0]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
# print("X_test shape: " + str(X_test.shape))
# print("Y_test shape: " + str(Y_test.shape))

image_x = 32
image_y = 32

# train_y = to_categorical(Y_train)
# test_y = to_categorical(Y_test)
# train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
# test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
X_train = x_train.reshape(x_train.shape[0], image_x, image_y, 1)
X_test = x_test.reshape(x_test.shape[0], image_x, image_y, 1)


# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(train_y.shape))


# Building a model


def keras_model(image_x, image_y):
    n_classes = 46
    model = Sequential()
    kernelSize = (3, 3)
    ip_activation = 'relu'
    in_shape = (image_x, image_y, 1)
    model.add(Conv2D(filters=32, kernel_size=kernelSize, input_shape=in_shape, activation=ip_activation))
    model.add(Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))  # Add the Pooling layer
    model.add(Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation))
    model.add(Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))  # Add the Pooling layer
    model.add(Flatten())
    # Now add the Dense layers
    model.add(Dense(units=128, activation=ip_activation, kernel_initializer='uniform'))
    # Let's add one more before proceeding to the output layer
    model.add(Dense(units=64, activation=ip_activation, kernel_initializer='uniform'))
    model.add(Dense(units=n_classes, activation='softmax', kernel_initializer='uniform'))
    # Compile the classifier using the configuration we want
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint1 = ModelCheckpoint("devanagari.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    return model, callbacks_list


model, callbacks_list = keras_model(image_x, image_y)
model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=5, batch_size=64, callbacks=callbacks_list)
scores = model.evaluate(X_test, test_y, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
print_summary(model)
model.save('devanagari.h5')
