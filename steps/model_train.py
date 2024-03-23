import os
import joblib
from sklearn.base import RegressorMixin
from src.build_model import RandomForestModel, KNeighborsRegressorModel, DecisionTreeRegressorModel

def train_model_and_evaluate_and_save_the_best(X_train, X_test, y_train, y_test) -> RegressorMixin:
    """
    Trains multiple models, evaluates them, and saves the best model.

    Args:
        X_train (pd.DataFrame): The training features.
        X_test (pd.DataFrame): The testing features.
        y_train (pd.Series): The training targets.
        y_test (pd.Series): The testing targets.

    Returns:
        dict: The dictionary containing the best models.
    """

    # Instantiation of models
    random_forest_model = RandomForestModel()
    decision_tree_model = DecisionTreeRegressorModel()
    knn_model = KNeighborsRegressorModel()

    # Training and evaluation of models
    trained_random_forest_model = random_forest_model.train(X_train, y_train)
    rf_score = random_forest_model.evaluate(trained_random_forest_model, X_test, y_test)

    trained_decision_tree_model = decision_tree_model.train(X_train, y_train)
    dt_score = decision_tree_model.evaluate(trained_decision_tree_model, X_test, y_test)

    trained_knn_model = knn_model.train(X_train, y_train)
    knn_score = knn_model.evaluate(trained_knn_model, X_test, y_test)

    # Saving the best models
    best_models = {}
    if (rf_score > dt_score) and (rf_score > knn_score):
        best_models['random_forest'] = trained_random_forest_model
    elif dt_score > knn_score:
        best_models['decision_tree'] = trained_decision_tree_model
    else:
        best_models['knn'] = trained_knn_model

    print(f"Features of the training data: {X_train.columns}")

    # Saving the best models in separate files
    file_path = r"C:\Users\djamb\OneDrive - Université Centrale\ML PROJECTS\PREDICTION PRIX LOGEMENT\model_saved"

    for model_name, model in best_models.items():
        for i, (model_name, model) in enumerate(best_models.items()):
            # Generate the file name for the model with incremental number
            file_name = f"{model_name}_model_{i}.pkl"

            # Full path of the file
            file_full_path = os.path.join(file_path, file_name)

            # Save the model to the file
            joblib.dump(model, file_full_path)

    print(f"The best models: {best_models}")
