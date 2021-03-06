import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Activation
from keras.models import Model, Sequential
from keras import regularizers

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


def getEncoders(layer_sizes, input_size):
    encoders = []
    decoders = []

    j = len(layer_sizes) -2

    for i in range(len(layer_sizes)):
        size = layer_sizes[i]
        if j < 0 :
            decoder_size = input_size
        else:
            decoder_size = layer_sizes[j]

        if i == 0:
            encoder = Dense(size, activation='relu', activity_regularizer=regularizers.l1(10e-8), input_dim=input_size)
        else:
            encoder = Dense(size, activation='relu', activity_regularizer=regularizers.l1(10e-8))

        decoder = Dense(decoder_size, activation='relu', activity_regularizer=regularizers.l1(10e-8))

        encoders.append(encoder)
        decoders.append(decoder)
        j -= 1



    return encoders, decoders

def train_layer_wise(encoders, decoders):
    decoders.reverse()
    local_encoders = []
    local_decoders  = []
    for i in range(len(encoders)):
        model = Sequential()
        local_encoders = encoders[:i+1]
        local_decoders = decoders[:i+1]
        local_decoders.reverse()
        for k in range(len(encoders)):
            j = k + 1
            if k > i:
                break

            if k < i:
                encoders[k].trainable = False

            encoder = local_encoders[k]

            model.add(encoder)

        for k in range(len(encoders)):
            j = k + 1
            if k > i:
                break

            if k < i:
                encoders[k].trainable = False

            decoder = local_decoders[k]

            model.add(decoder)

        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train, x_train, epochs=2,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

    return model


layer_sizes = [1024, 512]

encoders, decoders = getEncoders(layer_sizes, 784)

model = train_layer_wise(encoders, decoders)


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