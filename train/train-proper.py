import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import threading

grid_size = 100

def generate_complex_training_data(num_samples, num_threads):
    data = []
    samples_per_thread = num_samples // num_threads

    def generate_samples_thread(thread_id):
        thread_data = []
        for _ in range(samples_per_thread):
            initial_configuration = np.random.randint(2, size=(grid_size, grid_size))
            random_patterns = np.random.randint(2, size=(grid_size, grid_size))
            initial_configuration = np.logical_or(initial_configuration, random_patterns)
            next_generation = np.copy(initial_configuration)

            for i in range(grid_size):
                for j in range(grid_size):
                    live_neighbors = np.sum(initial_configuration[max(0, i-1):min(grid_size, i+2), max(0, j-1):min(grid_size, j+2)]) - initial_configuration[i, j]

                    if initial_configuration[i, j] == 1 and (live_neighbors < 2 or live_neighbors > 3):
                        next_generation[i, j] = 0
                    elif initial_configuration[i, j] == 0 and live_neighbors == 3:
                        next_generation[i, j] = 1

            thread_data.append((initial_configuration, next_generation))
        
        lock.acquire()
        data.extend(thread_data)
        lock.release()

    lock = threading.Lock()
    threads = []

    for i in range(num_threads):
        thread = threading.Thread(target=generate_samples_thread, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return np.array(data)

num_samples = 1000
num_threads = 4
training_data = generate_complex_training_data(num_samples, num_threads)

model = models.Sequential([
    layers.Input(shape=(grid_size, grid_size, 1)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

epochs = 10
model.fit(
    training_data[:, 0, :, :, np.newaxis],
    training_data[:, 1, :, :, np.newaxis],
    epochs=epochs,
    verbose=2
)

model.save('./models/game_of_life_model_complex.h5')
