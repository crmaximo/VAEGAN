from keras.optimizers import *
import matplotlib.pyplot as plt
import numpy as np
from VAEGAN import decgen, batch_size, rows, columns, channel

noise = np.random.normal(0, 1, (batch_size, 256))
# optimizers
SGDop = SGD(lr=0.0003)
ADAMop = Adam(lr=0.0002)

G = decgen(5, 32, rows, columns, channel)
G.compile(optimizer=SGDop, loss='mse')
G.summary()

G.load_weights('generator.h5')
image = G.predict(noise)
image = np.uint8(image * 127.5 +127.5)
plt.imshow(image[0]), plt.show()

