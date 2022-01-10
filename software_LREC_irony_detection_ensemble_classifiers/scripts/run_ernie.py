import pandas as pd

from ernie import SentenceClassifier
from data_processing import remove_duplicates


# fine-tunes a bertweet model with given data
def train_bertweet(train_data, task_type, name, training_batch_size_=16, learning_rate_=2e-5):
    if task_type == 'A':
        labels_no = 2
    else:
        labels_no = 4

    tuples = [(el[2], el[1]) for el in train_data]
    tuples = remove_duplicates(tuples)
    df = pd.DataFrame(tuples)

    classifier = SentenceClassifier(model_name="vinai/bertweet-base", max_length=128, labels_no=labels_no)

    classifier.load_dataset(df, validation_split=0.15)
    classifier.fine_tune(epochs=4, learning_rate=learning_rate_, training_batch_size=training_batch_size_,
                         validation_batch_size=training_batch_size_ * 2)
    classifier.dump('../transformer_all/' + name)


# given the test-set and a path to an existing model, it predicts the labels for the test-set
def run_predict(test_data, task_type, modelname):
    if task_type == 'A':
        labels_no = 2
    else:
        labels_no = 4

    classifier = SentenceClassifier(model_path='../transformer_all/' + modelname, max_length=64, labels_no=labels_no)
    test_tuples = [(el[2], el[1]) for el in test_data]

    results = []
    for el in test_tuples:
        probabilities = classifier.predict_one(el[0])
        chosen = probabilities.index(max(list(probabilities)))
        gold = el[1]
        results.append([chosen, gold])

    return results
