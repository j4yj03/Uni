from sklearn import metrics
import numpy as np

class CV:
    @staticmethod
    def partition(vector, fold, k):
        size = vector.shape[0]
        start = (size/k)*fold
        end = (size/k)*(fold+1)
        validation = vector[start:end]
        if str(type(vector)) == "<class 'scipy.sparse.csr.csr_matrix'>":
            indices = range(start, end)
            mask = np.ones(vector.shape[0], dtype=bool)
            mask[indices] = False
            training = vector[mask]
        elif str(type(vector)) == "<type 'numpy.ndarray'>":
            training = np.concatenate((vector[:start], vector[end:]))
        return training, validation

    @staticmethod
    def Cross_Validation(learner, k, examples, labels):
        train_folds_score = []
        validation_folds_score = []
        for fold in range(0, k):
            training_set, validation_set = Cross_Validation.partition(examples, fold, k)
            training_labels, validation_labels = Cross_Validation.partition(labels, fold, k)
            
            learner.fit(training_set, training_labels)
            training_predicted = learner.predict(training_set)
            validation_predicted = learner.predict(validation_set)
            train_folds_score.append(metrics.accuracy_score(training_labels, training_predicted))
            validation_folds_score.append(metrics.accuracy_score(validation_labels, validation_predicted))
        return train_folds_score, validation_folds_score
