print("Hello world!")

import tensorflow as tf
import pygame as pg

print("\n\n##########################\nTensorFlow version:", tf.__version__)
print("PyGame version:",pg.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


def sep():
    print("\n\n##########################################\n")


sep()
predictions = model(x_train[:1]).numpy()
print("predictions")
print(predictions)

sep()
print("tf.nn.softmax(predictions).numpy()")
print(tf.nn.softmax(predictions).numpy())

sep()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print("loss_fn(y_train[:1], predictions).numpy()")
print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

