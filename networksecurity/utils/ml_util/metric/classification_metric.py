from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculate classification metrics (precision, recall, and F1-score) 
    and return them as a ClassificationMetricArtifact object.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        ClassificationMetricArtifact: Object containing precision, recall, and F1-score.
    
    Raises:
        NetworkSecurityException: Custom exception for any error in processing.
    """
    try:
        # Calculate classification metrics
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)

        # Create artifact with the calculated metrics
        classification_metric = ClassificationMetricArtifact(
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            f1_score=model_f1_score  # Ensure the artifact supports this field
        )
        return classification_metric

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
