import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

loaded_model = tf.keras.models.load_model('./train/models/game_of_life_model_complex.h5')

grid_size = 100

initial_configuration = np.random.randint(2, size=(grid_size, grid_size))

threshold = 0.8

plt.figure(figsize=(8, 8))

while True:
    plt.clf()
    plt.imshow(initial_configuration, cmap='binary')
    plt.title('Current Generation')
    plt.pause(0.1)

    initial_configuration_input = initial_configuration[np.newaxis, :, :, np.newaxis]
    predicted_generation = loaded_model.predict(initial_configuration_input)

    predicted_generation_binary = (predicted_generation > threshold).astype(int)

    initial_configuration = predicted_generation_binary[0, :, :, 0]
