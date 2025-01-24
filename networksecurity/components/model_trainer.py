import os
import sys
from urllib.parse import urlparse

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object, 
    load_object, 
    load_numpy_array_data, 
    evaluate_models
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

import dagshub

# Initialize DagsHub for MLflow tracking
dagshub.init(
    repo_owner="akashagalaveaaa1", 
    repo_name="Network-Security", 
    mlflow=True
)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric):
        """
        Logs the metrics and model to MLflow.
        """
        try:
            mlflow.set_registry_uri("https://dagshub.com/akashagalaveaaa1/Network-Security.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                # Log metrics
                mlflow.log_metric("f1_score", classification_metric.f1_score)
                mlflow.log_metric("precision", classification_metric.precision_score)
                mlflow.log_metric("recall", classification_metric.recall_score)

                # Log the model
                mlflow.sklearn.log_model(best_model, "model")
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name="best_model")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Trains various models, evaluates them, logs the best model using MLflow, and saves it.
        """
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 128, 256],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    'learning_rate': [0.1, 0.01, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                           models=models, param=params)

            # Get the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            # Train and test metrics
            y_train_pred = best_model.predict(X_train)
            train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            y_test_pred = best_model.predict(X_test)
            test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Track metrics and model in MLflow
            self.track_mlflow(best_model, train_metric)
            self.track_mlflow(best_model, test_metric)

            # Save the model
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
            save_object("final_model/model.pkl", obj=best_model)

            # Create and return ModelTrainerArtifact
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process.
        """
        try:
            # Load transformed training and testing data
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1], train_arr[:, -1],
                test_arr[:, :-1], test_arr[:, -1]
            )

            # Train and save the best model
            return self.train_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
