import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10

(X, y), _ = cifar10.load_data()

print(X.shape)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(32))

model.add(Dense(10))
model.add(Activation(tf.nn.sigmoid))

model.compile(loss=tf.losses.softmax_cross_entropy, optimizer="adam", metrics=["accuracy"])

model.fit(X, y, batch_size=32, validation_split=0.1)