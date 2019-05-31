# pip3 install tensorflow numpy matplotlib

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# tf.logging.set_verbosity(tf.logging.ERROR)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

layerZero = tf.keras.layers.Dense(units=1, input_shape=[1])  

model = tf.keras.Sequential([layerZero])

model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))


history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")


plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

print(model.predict([111.0]))

# print("These are the layer variables: {}".format(l0.get_weights()))
