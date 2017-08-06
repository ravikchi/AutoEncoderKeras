import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Activation
from keras.models import Model, Sequential

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

input_img = Input(shape=(784,))

model = Sequential()
encoder1 = Dense(128, activation='relu', input_dim=784)
model.add(encoder1)
decoder1 = Dense(784, activation='sigmoid')
model.add(decoder1)

model.compile(optimizer='adadelta', loss='binary_crossentropy')

model.fit(x_train, x_train, epochs=80,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoder1.trainable = False
decoder1.trainable = False

model = Sequential()
model.add(encoder1)

encoder2 = Dense(64, activation='relu')
model.add(encoder2)

decoder2 = Dense(128, activation='relu')
model.add(decoder2)

model.add(decoder1)

model.compile(optimizer='adadelta', loss='binary_crossentropy')

model.fit(x_train, x_train, epochs=80,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoder2.trainable = False
decoder2.trainable = False

model = Sequential()

model.add(encoder1)
model.add(encoder2)

encoder3 = Dense(32, activation='relu')
model.add(encoder3)

decoder3 = Dense(64, activation='relu')
model.add(decoder3)

model.add(decoder2)
model.add(decoder1)

model.compile(optimizer='adadelta', loss='binary_crossentropy')

model.fit(x_train, x_train, epochs=80,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_imgs = model.predict(x_test)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()