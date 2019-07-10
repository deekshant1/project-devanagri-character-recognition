from keras.callbacks import ModelCheckpoint
from keras.utils import print_summary
import pandas as pd
# Scikit learn for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Keras Imports - CNN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

data = pd.read_csv(r"C:\Users\hp\Downloads\devanagari-character-set\data.csv")


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


image_x = 32
image_y = 32

X_train = x_train.reshape(x_train.shape[0], image_x, image_y, 1)
X_test = x_test.reshape(x_test.shape[0], image_x, image_y, 1)


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
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint1 = ModelCheckpoint("devanagari.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    return model, callbacks_list


model, callbacks_list = keras_model(image_x, image_y)
model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=5, batch_size=64, callbacks=callbacks_list)
scores = model.evaluate(X_test, test_y, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
print_summary(model)
model.save('devanagari.h5')
