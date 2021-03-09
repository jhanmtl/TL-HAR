import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import os


def load_subjects(npy_dir,
                  seen_subj,
                  seen_device,
                  seen_sensor,
                  unseen_subj,
                  unseen_device,
                  unseen_sensor,
                  act_int_lookup,
                  act_subset=None):
    seen_x = np.zeros((1, 100, 3))
    seen_y = np.zeros(1)

    for s in seen_subj:
        x, y = load_npy(npy_dir, s, seen_device, seen_sensor)
        seen_x = np.concatenate((seen_x, x), axis=0)
        seen_y = np.concatenate((seen_y, y))

    seen_x = seen_x[1:, :, :]
    seen_y = seen_y[1:]

    unseen_x, unseen_y = load_npy(npy_dir, unseen_subj, unseen_device, unseen_sensor)

    if act_subset is not None:
        int_subset = [act_int_lookup[i] for i in act_subset]

        seen_mask = [i in int_subset for i in seen_y]
        unseen_mask = [i in int_subset for i in unseen_y]

        seen_x = seen_x[seen_mask]
        seen_y = seen_y[seen_mask]

        unseen_x = unseen_x[unseen_mask]
        unseen_y = unseen_y[unseen_mask]

        old_labels = np.unique(seen_y)
        label_dict = {old_labels[i]: i for i in range(len(old_labels))}

        remapped_seen_y = [label_dict[i] for i in seen_y]
        remapped_unseen_y = [label_dict[i] for i in unseen_y]

        seen_y = np.array(remapped_seen_y)
        unseen_y = np.array(remapped_unseen_y)

    num_class = len(np.unique(seen_y))
    return (seen_x, seen_y), (unseen_x, unseen_y), num_class


def load_npy(datadir, subj, device, sensor):
    datapath = os.path.join(datadir, "{}_{}_{}_data.npy".format(subj, device, sensor))
    labelpath = os.path.join(datadir, "{}_{}_{}_label.npy".format(subj, device, sensor))

    x = np.load(datapath)
    y = np.load(labelpath)

    return x, y


def plot_pca_distributions(extractor, source_x, source_y, target_x, target_y,num_class):
    all_x = np.concatenate((source_x, target_x), axis=0)
    extracted_x = extractor.predict(all_x)

    pca = PCA(n_components=2)

    x_decomposed = pca.fit_transform(extracted_x)

    decomposed_df = pd.DataFrame()
    decomposed_df['x'] = x_decomposed[:, 0]
    decomposed_df['y'] = x_decomposed[:, 1]
    decomposed_df['activity'] = np.concatenate((source_y, target_y))
    decomposed_df['source domain'] = [True] * len(source_y) + [False] * len(target_y)

    plt.figure(figsize=(32, 16))
    sns.scatterplot(x='x', y='y', hue='activity',
                    data=decomposed_df,
                    palette=sns.color_palette('Set2', num_class),
                    style='source domain',
                    s=100,
                    edgecolor=['k' if was_seen else 'b' for was_seen in decomposed_df['source domain']],
                    )
