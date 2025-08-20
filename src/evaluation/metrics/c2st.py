import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt

def compute_c2st(inference_samples, reference_samples, test_size, random_state, plot=True, obs_idx=None):
    """
    Computes the classifier two-sample test score
    and (optional) plots inference vs reference samples.
    """
    x = np.concatenate((inference_samples, reference_samples), axis=0)
    y = np.concatenate([np.zeros(len(inference_samples)), np.ones(len(reference_samples))])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    classifier = LogisticRegression(max_iter=10000)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)


    if plot and inference_samples.shape[1] == 2: #plots are only generated for samples with correct shape
        plt.figure(figsize=(6, 6))
        plt.scatter(inference_samples[:, 0], inference_samples[:, 1],
                        alpha=0.5, label="Posterior (inference)")
        plt.scatter(reference_samples[:, 0], reference_samples[:, 1],
                        alpha=0.5, label="Reference posterior")
        plt.xlabel("θ₁")
        plt.ylabel("θ₂")
        title = f"C2ST={accuracy:.3f}"
        if obs_idx is not None:
            title += f" (observation {obs_idx})"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return accuracy
