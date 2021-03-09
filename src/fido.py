import tensorflow as tf
import numpy as np


class FIDO():
    def __init__(self, batchsize=1, epochs=15, alpha=5, classifier_lr=1e-3, extractor_lr=1e-3, reconstructor_lr=1e-3):
        self.alpha = alpha
        self.batchsize = batchsize
        self.epochs = epochs

        self.classifier_lr = classifier_lr
        self.extractor_lr = extractor_lr
        self.reconstructor_lr = reconstructor_lr

        self.classifier_opt = tf.keras.optimizers.Adam(self.classifier_lr)
        self.extractor_opt = tf.keras.optimizers.Adam(self.extractor_lr)
        self.reconstructor_opt = tf.keras.optimizers.Adam(self.reconstructor_lr)

        self.categorical_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.distance_loss = tf.keras.losses.Huber()

        self.class_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.class_loss = tf.keras.metrics.Mean()
        self.rec_loss = tf.keras.metrics.Mean()

        self.extractor = None
        self.classifier = None
        self.reconstructor = None

        self.class_predictor = None
        self.data_reconstructor = None

    def set_submodels(self, submodel_dict):
        self.extractor = submodel_dict['extractor']
        self.classifier = submodel_dict['classifier']
        self.reconstructor = submodel_dict['reconstructor']

        self.class_predictor = tf.keras.models.Sequential([self.extractor, self.classifier])
        self.data_reconstructor = tf.keras.models.Sequential([self.extractor, self.reconstructor])

    def in_domain_train(self, seen_x, seen_y):
        self.class_predictor.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                     optimizer=tf.keras.optimizers.Adam(self.classifier_lr),
                                     metrics=['acc'])
        self.class_predictor.fit(seen_x,
                                 seen_y,
                                 epochs=self.epochs,
                                 batch_size=self.batchsize
                                 )

    def cross_domain_train(self, seen_x, seen_y, unseen_x):
        TRAIN_STEPS_PER_EPOCH = min(int(len(seen_x) / self.batchsize), int(len(unseen_x) / self.batchsize)) + 1

        seen_dataset = tf.data.Dataset.from_tensor_slices((seen_x, seen_y))
        seen_dataset = seen_dataset.shuffle(buffer_size=len(seen_x) + 1).batch(self.batchsize)

        unseen_dataset = tf.data.Dataset.from_tensor_slices(unseen_x)
        unseen_dataset = unseen_dataset.shuffle(buffer_size=len(unseen_x) + 1).batch(self.batchsize)

        for epoch in range(self.epochs):
            print("epoch {}/{}".format(epoch + 1, self.epochs))
            tbar = tf.keras.utils.Progbar(target=TRAIN_STEPS_PER_EPOCH, unit_name="batch", width=30)

            for count, ((source_x, source_y), target_x) in enumerate(zip(seen_dataset, unseen_dataset)):
                self._cross_domain_train_step(source_x, source_y, target_x)

                progbar_classifier_acc = self.class_acc.result().numpy()
                progbar_classifier_loss = self.class_loss.result().numpy()
                progbar_rec_loss = self.rec_loss.result().numpy()

                progbar_values = [('classifier acc', progbar_classifier_acc),
                                  ('classifier loss', progbar_classifier_loss),
                                  ('mae loss', progbar_rec_loss)
                                  ]

                tbar.update(count, values=progbar_values)
            tbar.update(count + 1, values=progbar_values, finalize=True)

    def classify(self, x):
        pred_scores = self.class_predictor.predict(x)
        pred_classes = np.argmax(pred_scores, axis=-1)
        return pred_classes

    def evaluate_classifier_acc(self, x, ytrue):
        ypred = self.classify(x)
        acc = np.mean(ypred == ytrue)
        return acc

    @tf.function
    def _cross_domain_train_step(self, source_x, source_y, target_x):
        self._reset_metrics()

        combined_x = tf.concat((source_x, target_x), axis=0)
        combined_x = tf.random.shuffle(combined_x)

        with tf.GradientTape(persistent=True) as tape:
            class_pred = self.class_predictor(source_x, training=True)
            class_loss = self.categorical_loss(source_y, class_pred)

            reconstructed_x = self.data_reconstructor(combined_x)
            reconstruction_loss = self.distance_loss(reconstructed_x, combined_x)
            reconstruction_loss *= self.alpha

        classifier_gradient = tape.gradient(class_loss, self.classifier.trainable_variables)
        reconstructor_gradient = tape.gradient(reconstruction_loss, self.reconstructor.trainable_variables)

        extractor_gradient_1 = tape.gradient(reconstruction_loss, self.extractor.trainable_variables)
        extractor_gradient_2 = tape.gradient(class_loss, self.extractor.trainable_variables)

        self.classifier_opt.apply_gradients(zip(classifier_gradient, self.classifier.trainable_variables))
        self.reconstructor_opt.apply_gradients(zip(reconstructor_gradient, self.reconstructor.trainable_variables))

        self.extractor_opt.apply_gradients(zip(extractor_gradient_1, self.extractor.trainable_variables))
        self.extractor_opt.apply_gradients(zip(extractor_gradient_2, self.extractor.trainable_variables))

        self.class_acc(source_y, class_pred)
        self.class_loss(class_loss)
        self.rec_loss(reconstruction_loss)

    def _reset_metrics(self):
        self.class_acc.reset_states()
        self.class_loss.reset_states()
        self.rec_loss.reset_states()
