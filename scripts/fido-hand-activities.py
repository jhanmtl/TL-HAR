import dataset as data
import tensorflow as tf
import fido
import model_configs as models
import os
import json


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

seen_subj = list(range(1600, 1603))
unseen_subj = 1610

seen_device = "watch"
unseen_device = "watch"

seen_sensor = "accel"
unseen_sensor = "accel"

alpha = 1
batchsize = 64
EPOCHS = 3
cv = 5

clf_lr = 1e-4
rec_lr = 1e-4
ext_lr = 1e-4

act_subset = ['Dribblinlg (Basketball)',
              'Playing Catch w/Tennis Ball',
              'Typing',
              'Writing',
              'Clapping',
              'Brushing Teeth',
              'Folding Clothes'
              ]

num_class = len(act_subset)
datadir = os.path.join(os.path.abspath(os.path.pardir), "data")
npy_dir = os.path.join(datadir, "npy")

with open(os.path.join(datadir, "act_to_int.json"), "r") as jpath:
    act_int_lookup = json.load(jpath)

(seen_x, seen_y), (unseen_x, unseen_y), num_class = data.load_subjects(npy_dir,
                                                                       seen_subj,
                                                                       seen_device,
                                                                       seen_sensor,
                                                                       unseen_subj,
                                                                       unseen_device,
                                                                       unseen_sensor,
                                                                       act_int_lookup,
                                                                       act_subset)

print(seen_x.shape, seen_y.shape)
print(unseen_x.shape, unseen_y.shape)

crossdomain_source_acc = []
crossdomain_target_acc = []

for c in range(cv):
    tf.keras.backend.clear_session()
    crossdomain_model = fido.FIDO(batchsize=batchsize,
                                  epochs=EPOCHS,
                                  alpha=alpha,
                                  classifier_lr=clf_lr,
                                  extractor_lr=ext_lr,
                                  reconstructor_lr=rec_lr)

    crossdomain_model.set_submodels(models.submodel_config_march_09(num_class))
    crossdomain_model.cross_domain_train(seen_x, seen_y, unseen_x)

    source_acc = crossdomain_model.evaluate_classifier_acc(seen_x, seen_y)
    target_acc = crossdomain_model.evaluate_classifier_acc(unseen_x, unseen_y)
    print("========================== cv {}/{} ==========================".format(c + 1, cv))
    print(source_acc)
    print(target_acc)
    print("==============================================================")

    crossdomain_source_acc.append(source_acc)
    crossdomain_target_acc.append(target_acc)
