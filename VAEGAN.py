from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
import os
from keras import metrics, backend as K
from PIL import Image
import numpy as np

filenames = list()
labels = list()

images = os.path.join("img_align_celeba_png/")
annot = os.path.join("list_attr_celeba_png.txt")

with open(annot) as in_file:
    count_datapoints = int(in_file.readline())
    plaintext_labels = in_file.readline().split()

    for line in in_file:
        splitted = line.split()

        filenames.append(os.path.join(images, splitted[0]))

        properties_celebrity = [float(x) for x in splitted[1:]]
        properties_celebrity = [max(0.0, x) for x in properties_celebrity]
        labels.append(properties_celebrity)
assert len(filenames) == len(labels)

# print(labels[:4])
# print(filenames[:4])

show_number = len(filenames)
datasetlist = []
# print(plaintext_labels)
for filename, properties in zip(filenames[:show_number], labels[:show_number]):
    image = Image.open(os.path.join(filename))
    image = image.resize([64, 64], Image.ANTIALIAS)
    image = (np.asarray(image) - 127.5 / 127.5)
    #	image = plt.imread(image)
    #	plt.imshow(image)
    # plt.show()
    datasetlist.append(image)
    print(filename)
    # print(properties)

dataset = np.asarray(datasetlist)
print(dataset.shape)

def encoder(kernel, filter, rows, columns, channel):
    X = Input(shape=(rows, columns, channel))
    model = Conv2D(filters=filter, kernel_size=kernel, strides=2, padding='same')(X)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv2D(filters=filter*2, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv2D(filters=filter*4, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv2D(filters=filter*8, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Flatten()(model)

    mean = Dense(512)(model)
    logsigma = Dense(512, activation='tanh')(model)
    meansigma = Model([X], [mean, logsigma])
    return meansigma


def decgen(kernel, filter, rows, columns, channel):
    X = Input(shape=(2048,))

    model = Dense(filter*8*rows*columns)(X)
    model = Reshape((rows, columns, filter * 8))(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = Activation('relu')(model)


    model = Conv2DTranspose(filters=filter*4, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = Activation('relu')(model)

    model = Conv2DTranspose(filters=filter*2, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = Activation('relu')(model)

    model = Conv2DTranspose(filters=filter, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = Activation('relu')(model)

    model = Conv2DTranspose(filters=channel, kernel_size=kernel, strides=2, padding='same')(model)
    model = Activation('tanh')(model)

    model = Model(X, model)
    return model


def discriminator(kernel, filter, rows, columns, channel):
    X = Input(shape=(rows, columns, channel))

    model = Conv2D(filters=filter*2, kernel_size=kernel, strides=2, padding='same')(X)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv2D(filters=filter*4, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv2D(filters=filter*8, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv2D(filters=filter*8, kernel_size=kernel, strides=2, padding='same')(model)


    dec = BatchNormalization(epsilon=1e-5)(model)
    dec = LeakyReLU(alpha=0.2)(dec)
    dec = Flatten()(dec)
    dec = Dense(1, activation='sigmoid')(dec)

    output = Model([X], [dec, model])
    return output


batch_size = 512
rows = 64
columns = 64
channel = 3
epochs = 20000
datasize = len(dataset)
noise = np.random.normal(0, 1, (batch_size, 256))
# optimizers
SGDop = SGD(lr=0.0003)
ADAMop = Adam(lr=0.0002)
# encoder
E = encoder(5, 32, rows, columns, channel)
E.compile(optimizer='sgd', loss='mse')
E.summary()
# generator/decoder
G = decgen(5, 32, rows, columns, channel)
G.compile(optimizer=SGDop, loss='mse')
G.summary()
# discriminator
D = discriminator(5, 32, rows, columns, channel)
D.compile(optimizer=SGDop, loss='mse')
D.summary()
D_fixed = discriminator(5, 32, rows, columns, channel)
D_fixed.compile(optimizer=SGDop, loss='mse')
# VAE
X = Input(shape=(rows, columns, channel))
latent_rep = E(X)[0]
output = G(latent_rep)
VAE = Model(X, output)
Img = D.input
E_mean, E_logsigma = E(Img)
crossent = 64 * metrics.mse(K.flatten(X), K.flatten(output))
Z = G.input
Z2 = Input(512)
Img = D.input
G_train = G(Z)
G_dec = G(E_mean + Z2 * E_logsigma)
D_fake, F_fake = D(G_dec)
D_true, F_true = D(Img)
kl= - 0.5 * K.sum(1 + E_logsigma - K.square(E_mean) - K.exp(E_logsigma), axis=-1)
VAEloss = K.mean(crossent + kl)


VAE.add_loss(VAEloss)
VAE.compile(optimizer=SGDop)

for epoch in range(epochs):
    latent_vect = E.predict(dataset)[0]
    encImg = G.predict(latent_vect)
    fakeImg = G.predict(noise)

    DlossTrue = D.train_on_batch(dataset, np.ones((batch_size, 1)))
    DlossEnc = D.train_on_batch(encImg, np.ones((batch_size, 1)))
    DlossFake = D.train_on_batch(encImg, np.zeros((batch_size, 1)))

    cnt = epoch
    while cnt > 3:
        cnt = cnt - 4

    if cnt == 0:
        GlossEnc = G.train_on_batch(latent_vect, np.ones((batch_size, 1)))
        GlossGen = G.train_on_batch(noise, np.ones((batch_size, 1)))
        Eloss = VAE.train_on_batch(dataset, None)

    chk = epoch

    while chk > 50:
        chk = chk - 51

    if chk == 0:
        D.save_weights('discriminator.h5')
        G.save_weights('generator.h5')
        E.save_weights('encoder.h5')

    print("epoch number", epoch + 1)
    print("loss:")
    print("D:", DlossTrue, DlossEnc, DlossFake)
    print("G:", GlossEnc, GlossGen)
    print("VAE:", Eloss)

print('Training done,saving weights')
D.save_weights('discriminator.h5')
G.save_weights('generator.h5')
E.save_weights('encoder.h5')
print('end')
