import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
print(x_train.shape)
print(y_train.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs")

model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard])

# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss)
# print(val_acc)
#
# model.save('epic_num_reader.model')
#
# new_model = tf.keras.models.load_model('epic_num_reader.model')
#
# predictions = new_model.predict(x_test)
# print(predictions)
#
# print(np.argmax(predictions[0]))
#
# plt.imshow(x_test[0], cmap=plt.cm.binary)
# plt.show()
