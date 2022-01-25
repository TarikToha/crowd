import pickle

from keras.models import model_from_json
from keras.utils import plot_model
from matplotlib import pyplot as plt

bench_name = 'ucf_18_dense'

# include loss weights
reg = 0.1
cls = 1

hist_path = 'history/custom_cnn_reg_10cls_' + bench_name + '.hist'

NAME = hist_path.split('.')[0]

with open("{}_model.json".format(NAME)) as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("{}_best.weights".format(NAME))

loss = {
    'regression_output': 'mean_squared_error',
    'classification_output': 'sparse_categorical_crossentropy'
}

loss_weights = {
    'regression_output': reg,
    'classification_output': cls
}

metrics = {
    'regression_output': 'mean_absolute_error',
    'classification_output': 'accuracy'
}

model.compile(optimizer='adam', loss=loss, loss_weights=loss_weights, metrics=metrics)

model.save("{}_best.model".format(NAME))

plot_model(model, to_file="{}_model.pdf".format(NAME), show_shapes=True)

with open(hist_path, "rb") as hist:
    history = pickle.load(hist)

# print(history.keys())

fig = plt.figure()

# summarize history for accuracy
fig.add_subplot(2, 1, 1)
plt.plot(history['regression_output_mean_absolute_error'])
plt.plot(history['val_regression_output_mean_absolute_error'])
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

fig.add_subplot(2, 1, 2)
plt.plot(history['classification_output_accuracy'])
plt.plot(history['val_classification_output_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

fig.tight_layout(pad=0.5)
plt.savefig("{}_chart.pdf".format(NAME))
plt.show()
