import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

input_img = Input(shape=(784,))

encoded1 = Dense(128, activation='relu')(input_img)
decoded1 = Dense(784, activation='relu')(encoded1)

layer1 = Model(input_img, decoded1)

layer1.compile(optimizer='adadelta', loss='binary_crossentropy')

encoder1 = Model(input_img, encoded1)
encoded1_input = Input(shape=(128,))
decoded1_layer = layer1.layers[-1]

decoder1 = Model(encoded1_input, decoded1_layer(encoded1_input))

layer1.fit(x_train, x_train,
                epochs=80,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

x_train2 = encoder1.predict(x_train)
x_test2 = encoder1.predict(x_test)

encoded2 = Dense(64, activation='relu')(encoded1_input)
decoded2 = Dense(128, activation='relu')(encoded2)

layer2 = Model(encoded1_input, decoded2)

layer2.compile(optimizer='adadelta', loss='binary_crossentropy')

encoder2 = Model(encoded1_input, encoded2)
encoded2_input = Input(shape=(64,))
decoded2_layer = layer2.layers[-1]

decoder2 = Model(encoded2_input, decoded2_layer(encoded2_input))

layer2.fit(x_train2, x_train2,
                epochs=80,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test2, x_test2))

encoded1_imgs = encoder1.predict(x_test)
encoded2_imgs = encoder2.predict(encoded1_imgs)
decoded2_imgs = decoder2.predict(encoded2_imgs)
decoded_imgs = decoder1.predict(decoded2_imgs)

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

