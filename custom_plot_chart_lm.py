import pickle

import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model

bench_name = 'ucf_18_hajj'
version = 'v6'
lm_loss = 'bcep'
optimizer = 'sgd'
INPUT_SIZE = 256
DM_FACTOR = 8  # 256/32
batch_size = 16

postfix = lm_loss
hist_path = 'multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + '_' + \
            str(DM_FACTOR) + '_' + str(batch_size) + '_' + bench_name + '.hist'

NAME = hist_path.split('.hist')[0]

with open("{}_model.json".format(NAME)) as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("{}_best.weights".format(NAME))

model.save("{}_best.model".format(NAME), save_format='h5')

plot_model(model, to_file="{}_model.png".format(NAME), show_shapes=True)

with open(hist_path, "rb") as hist:
    history = pickle.load(hist)

# print(history.keys())

fig = plt.figure()
# summarize loss
fig.add_subplot(2, 1, 1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize metric
fig.add_subplot(2, 1, 2)
plt.plot(history['mean_absolute_error_count'])
plt.plot(history['val_mean_absolute_error_count'])
plt.ylabel('mean_absolute_error_count')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

fig.tight_layout(pad=0.5)
plt.savefig("{}_chart.pdf".format(NAME))
plt.show()
