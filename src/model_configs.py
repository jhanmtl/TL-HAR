import tensorflow as tf


def submodel_config_march_09(num_class):
    extractor = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(100, 3)),
                                            tf.keras.layers.Conv1D(516, 3, padding='causal', activation='relu'),
                                            tf.keras.layers.Conv1D(2048, 3, padding='causal', activation='relu'),

                                            tf.keras.layers.Flatten(),

                                            tf.keras.layers.Dense(512, activation='relu'),
                                            tf.keras.layers.Dense(512, activation='relu'),
                                            tf.keras.layers.Dense(1024, activation='relu'),
                                            tf.keras.layers.Dense(1024, activation='relu'),
                                            tf.keras.layers.Dense(1024, activation='relu'),
                                            ])

    classifier = tf.keras.models.Sequential([tf.keras.layers.Dense(512, activation='relu'),
                                             tf.keras.layers.Dense(num_class, activation='softmax')
                                             ])

    reconstructor = tf.keras.models.Sequential([tf.keras.layers.Dense(300),
                                                tf.keras.layers.Reshape((100, 3))
                                                ])

    m_dict = {"extractor": extractor,
              "classifier": classifier,
              "reconstructor": reconstructor}

    return m_dict


def make_resnet(k=3, f=256):
    data_in = tf.keras.layers.Input(shape=(100, 3), name='input')

    block_A = tf.keras.layers.Conv1D(filters=f, kernel_size=k, padding='causal')(data_in)
    block_A = tf.keras.layers.ReLU()(block_A)

    block_B = tf.keras.layers.Conv1D(filters=f, kernel_size=k, padding='causal')(block_A)
    block_B = tf.keras.layers.ReLU()(block_B)

    block_B = tf.keras.layers.Conv1D(filters=f, kernel_size=k, padding='causal')(block_B)
    block_B = tf.keras.layers.Add()([block_A, block_B])
    block_B = tf.keras.layers.ReLU()(block_B)

    downsample = tf.keras.layers.AveragePooling1D()(block_B)

    block_C = tf.keras.layers.Conv1D(filters=f, kernel_size=k, padding='causal')(downsample)
    block_C = tf.keras.layers.ReLU()(block_C)

    block_C = tf.keras.layers.Conv1D(filters=f, kernel_size=k, padding='causal')(block_C)
    block_C = tf.keras.layers.Add()([downsample, block_C])
    block_C = tf.keras.layers.ReLU()(block_C)

    downsample = tf.keras.layers.AveragePooling1D()(block_C)

    block_D = tf.keras.layers.Conv1D(filters=f, kernel_size=k, padding='causal')(downsample)
    block_D = tf.keras.layers.ReLU()(block_D)

    block_D = tf.keras.layers.Conv1D(filters=f, kernel_size=k, padding='causal')(block_D)
    block_D = tf.keras.layers.Add()([downsample, block_D])
    block_D = tf.keras.layers.ReLU()(block_D)

    downsample = tf.keras.layers.AveragePooling1D()(block_D)

    block_E = tf.keras.layers.Conv1D(filters=f, kernel_size=k, padding='causal')(downsample)
    block_E = tf.keras.layers.ReLU()(block_E)

    block_E = tf.keras.layers.Conv1D(filters=f, kernel_size=k, padding='causal')(block_E)
    block_E = tf.keras.layers.Add()([downsample, block_E])
    block_E = tf.keras.layers.ReLU()(block_E)

    head = tf.keras.layers.Flatten()(block_E)

    return tf.keras.Model(data_in, head)
