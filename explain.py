import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import shap

if __name__ == "__main__":
    models_directory = Path("models")

    with open(models_directory / "forty_pc_svm.pkl", "rb") as f:
        svm = pickle.load(f)

    training_data = np.load(models_directory / "training_data.npy")
    training_labels= np.loadtxt(models_directory / "training_labels.txt", dtype=str)
    test_data = np.load(models_directory / "test_data.npy")
    test_labels = np.loadtxt(models_directory / "test_labels.txt", dtype=str)

    explanations_directory = Path("explanations")
    
    K = 27
    kmeans_summary = shap.kmeans(training_data, K)
    print("Finished k-means summarization!")
    explainer = shap.KernelExplainer(svm.predict_proba, kmeans_summary, link="logit")

    shap_values = explainer.shap_values(test_data, nsamples=100)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], test_data[0],# link="logit",
            matplotlib=True, show=False)
    
    np.save(explanations_directory / "forty_pc_svm_explainer_expected_values.npy", explainer.expected_value)
    np.save(explanations_directory / "forty_pc_svm_shap_values.npy", shap_values)
    force_plot.savefig(explanations_directory / "force_plot_first_example.png", dpi=150, bbox_inches='tight')
