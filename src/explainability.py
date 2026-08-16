"""
Phase 7: Explainable AI (XAI) Module
Provides model interpretability via SHAP global feature importances, LIME local explanations, and temporal attention timeline visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ModelExplainer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def shap_analysis(self, model, X_train, X_test, feature_names=None):
        """
        Generate SHAP feature importance analysis and plot.
        :param model: Trained Classifier (SVM, RF, Logistic Regression)
        :param X_train: Training background data
        :param X_test: Test evaluation data
        :param feature_names: List of feature names
        :return: (top_feature_indices, mean_shap_values)
        """
        import shap

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

        try:
            # Use TreeExplainer for Random Forest, KernelExplainer/LinearExplainer for others
            if hasattr(model, "estimators_"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] # Class 1 (AD)
            else:
                bg_data = shap.sample(X_train, 30, random_state=42)
                explainer = shap.KernelExplainer(model.predict_proba, bg_data)
                shap_values = explainer.shap_values(X_test[:20])
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

            # Generate and save summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test[:len(shap_values)], feature_names=feature_names[:X_test.shape[1]], show=False)
            save_path = os.path.join(self.output_dir, "shap_summary.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"SHAP summary plot saved to {save_path}")

            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:10]
            return top_indices, mean_abs_shap
        except Exception as e:
            print(f"SHAP analysis notice: {e}. Generating fallback feature importance plot.")
            return self._fallback_feature_importance(model, X_train, feature_names)

    def _fallback_feature_importance(self, model, X_train, feature_names):
        """Fallback feature importance calculation using tree importances or linear coefficients."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.std(X_train, axis=0)

        top_indices = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=importances[top_indices], y=[feature_names[i] for i in top_indices], palette="viridis")
        plt.title("Feature Importance Analysis (XAI)")
        plt.xlabel("Relative Importance Score")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "shap_summary.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        return top_indices, importances

    def lime_explanation(self, model, X_train, sample, feature_names=None, class_names=["CN", "AD"]):
        """
        Generate LIME local explanation for a single prediction instance.
        """
        try:
            import lime
            import lime.lime_tabular

            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                feature_names=feature_names,
                class_names=class_names,
                mode="classification",
                discretize_continuous=True,
                random_state=42
            )

            exp = explainer.explain_instance(
                data_row=sample,
                predict_fn=model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                num_features=10
            )

            html_path = os.path.join(self.output_dir, "lime_explanation.html")
            exp.save_to_file(html_path)
            print(f"LIME local explanation saved to {html_path}")
            return exp
        except Exception as e:
            print(f"LIME explanation notice: {e}")
            return None

    def visualize_attention_timeline(self, attention_weights, time_timeline, save_path=None):
        """
        Visualize model attention weight over speech timeline.
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "attention_timeline.png")

        plt.figure(figsize=(10, 4))
        plt.plot(time_timeline, attention_weights, color="#2b5c8f", lw=2.5)
        plt.fill_between(time_timeline, attention_weights, alpha=0.3, color="#4c8bf5")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Gated Attention Weight")
        plt.title("Gated Cross-Attention Weight Across Utterance Timeline")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Attention timeline visualization saved to {save_path}")
