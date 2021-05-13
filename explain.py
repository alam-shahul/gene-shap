import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import shap

from vae.train import VAE

def calculate_shap_values(model, training_data, test_data, K=None):
    """Calculate SHAP values for the given model.

    Args:
        model: trained classification model
        training_data: data of shape (num_samples, num_features) on which model was trained
        test_data: data of shape (num_samples, num_features) on which model was tested 
        K: number of means to use for K-means summarization of training_data (used for
            calculating feature importances in SHAP values); if None, then all the
            training data are used for this calculation (potentially expensive)
    
    Returns:
        A tuple consisting of 1. the explainer's expected values and 2. the SHAP values.
    """

    if K:
        kmeans_summary = shap.kmeans(training_data, K)
        print("Finished k-means summarization!")
        explainer = shap.KernelExplainer(model, kmeans_summary, link="logit")
    else:
        explainer = shap.KernelExplainer(model, training_data, link="logit")
    
    shap_values = explainer.shap_values(test_data, nsamples=100)

    return explainer.expected_value, shap_values

def decompress_shap_values_pca(shap_values, inverse_projection):
    """Decompress PCA SHAP values in embedding space into gene-level equivalents.

    Args:
        shap_values: a matrix of SHAP values of shape
            (num_classes, num_samples, num_principle_components)

    Returns:
        Decompressed, approximate SHAP values for all genes in the original dataset (i.e. a
        matrix of shape (num_classes, num_samples, num_genes))
    """

    num_classes, num_samples, _ = shap_values.shape
    _, num_genes = inverse_projection.shape
    decompressed_shap_values = np.zeros((num_classes, num_samples, num_genes))
    for class_index, class_shap_values in enumerate(shap_values):
        decompressed_class_shap_values = class_shap_values @ inverse_projection
        decompressed_shap_values[class_index] = decompressed_class_shap_values

    return decompressed_shap_values

def calculate_deepshap_values(model, training_data, test_data, K=None):
    """Calculate deepSHAP values for the given model.

    Args:
        model: trained deep model
        training_data: data of shape (num_samples, num_features) on which model was trained
        test_data: data of shape (num_samples, num_features) on which model was tested 
        K: number of means to use for K-means summarization of training_data (used for
            calculating feature importances in SHAP values); if None, then all the
            training data are used for this calculation (potentially expensive)
    
    Returns:
        A tuple consisting of 1. the explainer's expected values and 2. the SHAP values.
    """

    if K:
        kmeans_summary = shap.kmeans(training_data, K)
        print("Finished k-means summarization!")
        explainer = shap.DeepExplainer(model, kmeans_summary, link="logit")
    else:
        explainer = shap.DeepExplainer(model, training_data, link="logit")
    
    shap_values = explainer.shap_values(test_data)

    return explainer.expected_value, shap_values

if __name__ == "__main__":
    models_directory = Path("models") / "full_dataset_l3"

    with open(models_directory / "forty_pc_svm.pkl", "rb") as f:
        svm = pickle.load(f)

    training_data = np.load(models_directory / "training_data.npy")
    training_labels= np.loadtxt(models_directory / "training_labels.txt", dtype=str, delimiter="\t")
    test_data = np.load(models_directory / "test_data.npy")
    test_labels = np.loadtxt(models_directory / "test_labels.txt", dtype=str, delimiter="\t")

    training_data = training_data[:len(training_data)//3]
    training_labels = training_data[:len(training_labels)//3]
    test_data = training_data[:len(test_data)//3]
    test_labels = training_data[:len(test_labels)//3]

    explanations_directory = Path("explanations")
    
    expected_value, shap_values = calculate_shap_values(svm.predict_proba, training_data, test_data, K=27)
    # force_plot = shap.force_plot(expected_value[0], shap_values[0][0,:], test_data[0],# link="logit",
    #         matplotlib=True, show=False)
    # 
    # np.save(explanations_directory / "forty_pc_svm_explainer_expected_values.npy", explainer.expected_value)
    # np.save(explanations_directory / "forty_pc_svm_shap_values.npy", shap_values)
    # force_plot.savefig(explanations_directory / "force_plot_first_example.png", dpi=150, bbox_inches='tight')

    shap_values = np.load(explanations_directory / "forty_pc_svm_shap_values.npy")
    inverse_projection = np.load("forty_ptp_inv_pt.npy")

    decompressed_shap_values = decompress_shap_values_pca(shap_values, inverse_projection.T)
    classes = list(svm.classes_)
    true_label_indices = [classes.index(label) for label in test_labels]
    decompressed_true_class_shap_values = decompressed_shap_values[true_label_indices, range(len(test_labels))]
    print(decompressed_true_class_shap_values.shape)
    np.save(explanations_directory / "forty_pc_svm_decompressed_true_shap_values.npy", decompressed_true_class_shap_values)
    # np.save(explanations_directory / "forty_pc_svm_decompressed_shap_values.npy", decompressed_shap_values)

    # models_directory = Path("models") / "full_dataset_l3_scvis"

    # with open(models_directory / "ten_dimension_scvis_svm.pkl", "rb") as f:
    #     svm = pickle.load(f)


    # training_data = np.load(models_directory / "training_data.npy")
    # training_labels= np.loadtxt(models_directory / "training_labels.txt", dtype=str, delimiter="\t")
    # test_data = np.load(models_directory / "test_data.npy")
    # test_labels = np.loadtxt(models_directory / "test_labels.txt", dtype=str, delimiter="\t")

    # training_data = training_data[:len(training_data)//3]
    # training_labels = training_data[:len(training_labels)//3]
    # test_data = training_data[:len(test_data)//3]
    # test_labels = training_data[:len(test_labels)//3]

    # explanations_directory = Path("explanations")
    # 
    # expected_value, shap_values = calculate_shap_values(svm.predict_proba, training_data, test_data, K=27)
    # np.save(explanations_directory / "ten_dimension_scvis_svm_explainer_expected_values.npy", explainer.expected_value)
    # np.save(explanations_directory / "ten_dimension_scvis_svm_shap_values.npy", shap_values)

    # expected_value, shap_values = calculate_shap_values(svm.predict_proba, training_data, test_data, K=27)
    # np.save(explanations_directory / "ten_dimension_scvis_svm_explainer_expected_values.npy", explainer.expected_value)
    # np.save(explanations_directory / "ten_dimension_scvis_svm_shap_values.npy", shap_values)
