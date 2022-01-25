import pickle

import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model

bench_name = 'shanghai_A'
version = 'v8.2'
dm_loss = 'msep'
lm_loss = 'bcep'
dm_weight = 1
lm_weight = 1
optimizer = 'adam'
INPUT_SIZE = 128
DM_FACTOR = 8  # 256/32
batch_size = 16

postfix = str(dm_weight) + '_' + dm_loss + '_' + str(lm_weight) + '_' + lm_loss
hist_path = 'multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + \
            '_' + str(DM_FACTOR) + '_' + str(batch_size) + '_' + bench_name + '.hist'

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
plt.plot(history['density_map_output_mean_absolute_error_count'])
plt.plot(history['val_density_map_output_mean_absolute_error_count'])
plt.ylabel('density_map_' + dm_loss)
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize metric
fig.add_subplot(2, 1, 2)
plt.plot(history['point_map_output_localized_counting_error'])
plt.plot(history['val_point_map_output_localized_counting_error'])
plt.ylabel('point_map_' + lm_loss)
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

fig.tight_layout(pad=0.5)
plt.savefig("{}_chart.pdf".format(NAME))
plt.show()
