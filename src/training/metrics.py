"""
Comprehensive Metrics for Deepfake Detection
Advanced evaluation metrics and monitoring utilities
From Hasif's Workspace
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)
import logging

logger = logging.getLogger(__name__)


class DeepfakeMetrics:
    """
    Comprehensive metrics calculator for deepfake detection
    """

    def __init__(self, metrics_list: Optional[List[str]] = None):
        """
        Initialize DeepfakeMetrics

        Args:
            metrics_list: List of metrics to calculate
        """
        self.metrics_list = metrics_list or [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
        ]

        # Available metrics
        self.available_metrics = {
            "accuracy": self._accuracy,
            "precision": self._precision,
            "recall": self._recall,
            "f1_score": self._f1_score,
            "auc_roc": self._auc_roc,
            "auc_pr": self._auc_pr,
            "specificity": self._specificity,
            "sensitivity": self._sensitivity,
            "mcc": self._matthews_correlation,
            "kappa": self._cohen_kappa,
            "balanced_accuracy": self._balanced_accuracy,
            "confusion_matrix": self._confusion_matrix,
        }

        logger.info(f"Initialized metrics: {self.metrics_list}")

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calculate all specified metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels or probabilities
            y_prob: Predicted probabilities (if y_pred are labels)
            threshold: Classification threshold

        Returns:
            Dictionary of calculated metrics
        """
        # Convert predictions to binary if they are probabilities
        if y_pred.max() <= 1.0 and y_pred.min() >= 0.0 and len(np.unique(y_pred)) > 2:
            y_prob = y_pred.copy()
            y_pred = (y_pred >= threshold).astype(int)
        elif y_prob is None:
            y_prob = y_pred.copy()

        results = {}

        for metric_name in self.metrics_list:
            if metric_name in self.available_metrics:
                try:
                    if metric_name in ["auc_roc", "auc_pr"]:
                        # These metrics need probabilities
                        results[metric_name] = self.available_metrics[metric_name](
                            y_true, y_prob
                        )
                    else:
                        # These metrics use binary predictions
                        results[metric_name] = self.available_metrics[metric_name](
                            y_true, y_pred
                        )
                except Exception as e:
                    logger.warning(f"Error calculating {metric_name}: {e}")
                    results[metric_name] = 0.0
            else:
                logger.warning(f"Unknown metric: {metric_name}")

        return results

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        return float(accuracy_score(y_true, y_pred))

    def _precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision"""
        return float(precision_score(y_true, y_pred, zero_division=0))

    def _recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall (sensitivity)"""
        return float(recall_score(y_true, y_pred, zero_division=0))

    def _f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score"""
        return float(f1_score(y_true, y_pred, zero_division=0))

    def _auc_roc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate AUC-ROC"""
        try:
            return float(roc_auc_score(y_true, y_prob))
        except ValueError:
            # Handle case where only one class is present
            return 0.5

    def _auc_pr(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate AUC-PR (Average Precision)"""
        try:
            return float(average_precision_score(y_true, y_prob))
        except ValueError:
            return 0.0

    def _specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if tn + fp == 0:
            return 0.0
        return float(tn / (tn + fp))

    def _sensitivity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate sensitivity (same as recall)"""
        return self._recall(y_true, y_pred)

    def _matthews_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Matthews Correlation Coefficient"""
        try:
            return float(matthews_corrcoef(y_true, y_pred))
        except ValueError:
            return 0.0

    def _cohen_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Cohen's Kappa"""
        try:
            return float(cohen_kappa_score(y_true, y_pred))
        except ValueError:
            return 0.0

    def _balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy"""
        sensitivity = self._sensitivity(y_true, y_pred)
        specificity = self._specificity(y_true, y_pred)
        return (sensitivity + specificity) / 2.0

    def _confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        return confusion_matrix(y_true, y_pred)

    def get_detailed_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get detailed classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            class_names: Names of classes

        Returns:
            Detailed report dictionary
        """
        if class_names is None:
            class_names = ["Real", "Fake"]

        # Calculate all metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if str(i) in report:
                per_class_metrics[class_name] = report[str(i)]

        return {
            "overall_metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": per_class_metrics,
            "classification_report": report,
        }


class MetricsTracker:
    """
    Track metrics over time during training
    """

    def __init__(self):
        """Initialize MetricsTracker"""
        self.history = {"train": [], "val": [], "test": []}
        self.best_metrics = {}

    def update(self, split: str, metrics: Dict[str, float], epoch: int):
        """
        Update metrics for a split

        Args:
            split: Data split (train, val, test)
            metrics: Metrics dictionary
            epoch: Current epoch
        """
        metrics_with_epoch = {"epoch": epoch, **metrics}
        self.history[split].append(metrics_with_epoch)

        # Update best metrics
        if split not in self.best_metrics:
            self.best_metrics[split] = metrics_with_epoch.copy()
        else:
            # Update if current accuracy is better
            if metrics.get("accuracy", 0) > self.best_metrics[split].get("accuracy", 0):
                self.best_metrics[split] = metrics_with_epoch.copy()

    def get_best_metrics(self, split: str) -> Dict[str, float]:
        """Get best metrics for a split"""
        return self.best_metrics.get(split, {})

    def get_latest_metrics(self, split: str) -> Dict[str, float]:
        """Get latest metrics for a split"""
        if self.history[split]:
            return self.history[split][-1]
        return {}

    def get_metric_history(self, split: str, metric: str) -> List[float]:
        """Get history of a specific metric"""
        return [entry.get(metric, 0) for entry in self.history[split]]


class ThresholdOptimizer:
    """
    Optimize classification threshold for best performance
    """

    def __init__(self, metric: str = "f1_score"):
        """
        Initialize ThresholdOptimizer

        Args:
            metric: Metric to optimize for
        """
        self.metric = metric
        self.metrics_calculator = DeepfakeMetrics()

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold for classification

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            thresholds: Thresholds to test (default: 0.1 to 0.9 in steps of 0.01)

        Returns:
            Tuple of (optimal_threshold, best_metrics)
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.01)

        best_threshold = 0.5
        best_score = 0.0
        best_metrics = {}

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            try:
                metrics = self.metrics_calculator.calculate_metrics(
                    y_true, y_pred, y_prob
                )
                score = metrics.get(self.metric, 0.0)

                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = metrics
            except Exception as e:
                logger.warning(
                    f"Error calculating metrics for threshold {threshold}: {e}"
                )
                continue

        return best_threshold, best_metrics

    def threshold_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Analyze performance across different thresholds

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            thresholds: Thresholds to analyze

        Returns:
            Dictionary with threshold analysis results
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)

        results = {
            "thresholds": thresholds.tolist(),
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "specificity": [],
        }

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            try:
                metrics = self.metrics_calculator.calculate_metrics(
                    y_true, y_pred, y_prob
                )

                results["accuracy"].append(metrics.get("accuracy", 0.0))
                results["precision"].append(metrics.get("precision", 0.0))
                results["recall"].append(metrics.get("recall", 0.0))
                results["f1_score"].append(metrics.get("f1_score", 0.0))
                results["specificity"].append(metrics.get("specificity", 0.0))
            except Exception as e:
                logger.warning(f"Error analyzing threshold {threshold}: {e}")
                # Append zeros for failed calculations
                for key in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "specificity",
                ]:
                    results[key].append(0.0)

        return results


# Utility functions
def calculate_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Calculate confidence intervals for a metric using bootstrap sampling

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_func: Function to calculate metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (metric_value, lower_bound, upper_bound)
    """
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        try:
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except:
            continue

    bootstrap_scores = np.array(bootstrap_scores)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    metric_value = metric_func(y_true, y_pred)
    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)

    return metric_value, lower_bound, upper_bound


def compare_models(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    metrics_list: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same dataset

    Args:
        y_true: True labels
        predictions_dict: Dictionary mapping model names to predictions
        metrics_list: List of metrics to calculate

    Returns:
        Dictionary with comparison results
    """
    if metrics_list is None:
        metrics_list = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]

    metrics_calculator = DeepfakeMetrics(metrics_list)
    results = {}

    for model_name, y_pred in predictions_dict.items():
        try:
            metrics = metrics_calculator.calculate_metrics(y_true, y_pred)
            results[model_name] = metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}")
            results[model_name] = {metric: 0.0 for metric in metrics_list}

    return results
