class TitanicModel(object):

    def __init__(self):
        pass

    def predict(self, test_x):
        pass

    def evaluate(self, predictions, results):
        tp, tn, fp, fn = 0, 0, 0, 0

        for prediction, result in zip(predictions, results):
            if prediction == 0:
                if result == 0:
                    tn += 1
                else:
                    fn +=1
            if prediction == 1:
                if result == 1:
                    tp += 1
                else:
                    fp += 1

        return (tp, tn, fp, fn)

    def accuracy(self, test_x, test_y):
        predictions = self.predict(test_x)
        conf_mat = self.evaluate(predictions, test_y)

        return (conf_mat[0] + conf_mat[1]) / (conf_mat[0] + conf_mat[1] + conf_mat[2] + conf_mat[3])

    def argmax(self, values):
        max = values[0]
        max_index = 0

        for i in range(len(values)):
            if values[i] > max:
                max_index = i
                max = values[i]

        return max_index
