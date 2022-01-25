from tensorflow.keras.models import model_from_json

bench_name = 'ucf_18_hajj'
version = 'v6'
lm_loss = 'bcep'
optimizer = 'adam'
INPUT_SIZE = 256
DM_FACTOR = 8  # 256/32
batch_size = 16

postfix = lm_loss
NAME = 'history_dm/multi_' + version + '_' + postfix + '_' + optimizer + '_' + str(INPUT_SIZE) + '_' + \
       str(DM_FACTOR) + '_' + str(batch_size) + '_' + bench_name

with open("{}_model.json".format(NAME)) as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

model.load_weights("{}_best.weights".format(NAME))
model.save("{}_best.model".format(NAME), save_format='h5')

model.load_weights("{}_temp.weights".format(NAME))
model.save("{}_temp.model".format(NAME), save_format='h5')
