import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_util.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from networksecurity.utils.ml_util.metric.classification_metric import get_classification_score
import mlflow


import dagshub
dagshub.init(repo_owner='akashagalaveaaa1', repo_name='Network-Security', mlflow=True)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        """
        Initializes the ModelTrainer with configurations and artifacts.
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def track_mlflow(self,best_model,classificationmetrics):
        with mlflow.start_run():
            f1_score=classificationmetrics.f1_score
            precision_score=classificationmetrics.precision_score
            recall_score=classificationmetrics.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)

            mlflow.sklearn.log_model(best_model,"model")



    def train_model(self, x_train, y_train, x_test, y_test):
        """
        Trains multiple models and selects the best one based on evaluation metrics.
        """
        try:
            # Define models and hyperparameters
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(algorithm='SAMME'),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss']
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7,  0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                }
            }

            # Evaluate models
            model_report = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Select the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Evaluate on training data
            y_train_pred = best_model.predict(x_train)
            train_metrics = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            ## track the experiments with mlflow
            self.track_mlflow(best_model,train_metrics)



            # Evaluate on testing data
            y_test_pred = best_model.predict(x_test)
            test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(best_model,test_metrics)



            # Save the model and preprocessor
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            ##model pusher

            save_object("final_models/model.pkl",best_model)

            # Prepare ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Prepares the training and testing data and initiates the model training process.
        """
        try:
            # Load transformed data
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_data = load_numpy_array_data(train_file_path)
            test_data = load_numpy_array_data(test_file_path)

            # Split into features and target
            x_train, y_train = train_data[:, :-1], train_data[:, -1]
            x_test, y_test = test_data[:, :-1], test_data[:, -1]

            # Train the model
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
