import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from pathlib import Path
import pickle

def filter_and_split_data(data, labels, desired_clusters=None):
    """Filters the data and labels that are desired.

    Args:
        data: (num_samples, num_genes)      
        labels: (num_samples, )
        desired_clusters: filters the desired clusters
    Returns:
        The splits of the data and labels as a four-element tuple.    
    """

    if desired_clusters:
        filter_index = np.isin(labels, desired_clusters)
    else:
        filter_index = np.full(labels.shape, True)

    filtered_data = data[filter_index]
    filtered_labels = labels[filter_index]

    training_data, test_data, training_labels, test_labels = train_test_split(
            filtered_data, filtered_labels, test_size=0.3, stratify=filtered_labels)

    return training_data, test_data, training_labels, test_labels

def train_SVM(training_data, training_labels):
    """Trains an SVM classifier (with scaling on the inputs).

    Args:
        training_data: the training data with shape (num_samples, num_features)
        labels: the labels with shape (num_samples, )
    Returns:
        A trained SVM that can be used for classification.
    """

    classifier = make_pipeline(StandardScaler(), SVC(gamma="auto", probability=True, verbose=True))
    classifier.fit(training_data, training_labels)

    return classifier

if __name__ == "__main__":
    data = np.load("pbma_forty_pcs.npy")
    labels = np.load("l1_labels.npy", allow_pickle=True)
    desired_clusters = ["NK", "B", "DC"]
    # desired_clusters = None

    training_data, test_data, training_labels, test_labels =  filter_and_split_data(data, labels, desired_clusters)
    print(training_data.shape)

    models_directory = Path("models")
    np.save(models_directory / "training_data.npy", training_data)
    np.save(models_directory / "test_data.npy", test_data)
    np.savetxt(models_directory / "training_labels.txt", training_labels, fmt="%s")
    np.savetxt(models_directory / "test_labels.txt", test_labels, fmt="%s")

    svm = train_SVM(training_data, training_labels)
    with open(models_directory / "forty_pc_svm.pkl", "wb") as f:
        pickle.dump(svm, f, pickle.HIGHEST_PROTOCOL)
    
    score = svm.score(test_data, test_labels)
    print("Top-1 Accuracy: {}".format(score))
