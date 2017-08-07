import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Activation
from keras.models import Model, Sequential, load_model
from keras import regularizers

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

input_img = Input(shape=(784,))

layer1_size = 1024
layer2_size = 512
layer3_size = 256

input_size = 784

num_epochs = 80

model = Sequential()
encoder1 = Dense(layer1_size, activation='relu', activity_regularizer=regularizers.l1(10e-8), input_dim=input_size)
model.add(encoder1)
decoder1 = Dense(input_size, activation='sigmoid')
model.add(decoder1)

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=num_epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoder1.trainable = False
decoder1.trainable = False

model = Sequential()
model.add(encoder1)

encoder2 = Dense(layer2_size, activation='relu',activity_regularizer=regularizers.l1(10e-8))
model.add(encoder2)

decoder2 = Dense(layer1_size, activation='relu')
model.add(decoder2)

model.add(decoder1)

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=num_epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoder2.trainable = False
decoder2.trainable = False

model = Sequential()

model.add(encoder1)
model.add(encoder2)

encoder3 = Dense(layer3_size, activation='relu',activity_regularizer=regularizers.l1(10e-8))
model.add(encoder3)

decoder3 = Dense(layer2_size, activation='relu')
model.add(decoder3)

model.add(decoder2)
model.add(decoder1)

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=num_epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


decoded_imgs = model.predict(x_test)

encoder1.trainable = True
encoder2.trainable = True
encoder3.trainable = True

decoder1.trainable = True
decoder2.trainable = True
decoder3.trainable = True

model = Sequential()

model.add(encoder1)
model.add(encoder2)
model.add(encoder3)

encoder4 = Dense(1, activation='relu')
model.add(encoder4)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=num_epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test))

metrics = model.evaluate(x_test, y_test)

for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))


# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

predictions = model.predict(x_test)
for i in range(10):
    print(y_test[i])
    print(predictions[i])