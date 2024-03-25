# Importing necessary libraries
# sklearn libraries for model building
from abc import ABC, abstractmethod
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
from mlflow.models import infer_signature

class BuildModel(ABC):
    """
    Abstract Base Class for building models.
    This class provides the template for building models.
    It includes methods to train and evaluate a model.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Abstract method to train the model.
        This method takes in the training data and labels.
        It should return the trained model.

        Parameters:
        X_train (array-like): Training data.
        y_train (array-like): Target variable for the training data.

        Returns:
            model: Trained model.
        """
        pass

    @abstractmethod
    def evaluate(self, y_pred, y_test):
        """
        Abstract method to evaluate the model.
        This method takes in the testing data and labels.
        It should return the evaluation score.

        Parameters:
        X_test (array-like): Testing data.
        y_test (array-like): Target variable for the testing data.

        Returns:
        score: Evaluation score of the model.
        """
        pass
    def predict(self, model, X_test):
        """
        Predict using the trained model.

        Parameters:
        ----------
        model : estimator
            Trained model.
        X : array-like
            Input data for prediction.

        Returns:
        -------
        predictions : array-like
            Predicted values.
        """
        pass
    
class RandomForestModel(BuildModel):

    def train(self, X_train, y_train):
        """
        Train the Random Forest Regressor model using Grid Search Cross Validation.
        
        Parameters:
        ----------
        X_train : array-like
            Training data.
        y_train : array-like
            Target variable for the training data.

        Returns:
        -------
        best_model : estimator
            Trained model.
        """
        model = RandomForestRegressor()
        params = {
            'n_estimators': [100, 200,300],  # number of trees in the forest
            'max_depth': [None, 10, 20,100,200]  # maximum depth of each tree
        }
        grid_search = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logging.info(f"Best parameters for RandomForestRegressor: {best_params}")
        return best_model
    
    def predict(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred

    def evaluate(self,y_pred, y_test):
        """
        Evaluate the trained model using R-squared score.

        Parameters:
        ----------
        model : estimator
            Trained model.
        X_test : array-like
            Testing data.
        y_test : array-like
            Target variable for the testing data.

        Returns:
        -------
        score : float
            R-squared score of the model.
        """
        score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        # Print and log the evaluation score
        print("-------Random Forest Model -------")
        logging.info(f"Model score: {score*100:.2f} -- Mean Squared Error: {mse:.2f} -- Mean Absolute Error: {mae:.2f} -- Max Error: {max_err:.2f}")
        return score, mse, mae, max_err
    


class DecisionTreeRegressorModel(BuildModel):

    def train(self, X_train, y_train):
        model = DecisionTreeRegressor()
        params = {
            'max_depth': [None, 5, 10,100,60,200,250,300],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter':  ['best', 'random'] 
        }
        grid_search = GridSearchCV(model, params, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logging.info(f"Best parameters for DecisionTreeRegressor: {best_params}")
        return best_model
    
    def predict(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred
    
    def evaluate(self, y_pred, y_test):
        score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        # Print and log the evaluation score
        print("-------Decision Tree Regressor Model-------")
        logging.info(f"Model score: {score*100:.2f} -- Mean Squared Error: {mse:.2f} -- Mean Absolute Error: {mae:.2f} -- Max Error: {max_err:.2f}")
        return score, mse, mae, max_err
    



class KNeighborsRegressorModel(BuildModel):

    def train(self, X_train, y_train):
        """
        Train the KNeighborsRegressor model using Grid Search Cross Validation.
        
        Parameters:
        ----------
        X_train : array-like
            Training data.
        y_train : array-like
            Target variable for the training data.

        Returns:
        -------
        best_model : estimator
            Trained model.
        """
        # Define model and parameters for grid search
        model = KNeighborsRegressor()
        params = {
            'n_neighbors': [3, 5, 7,10],  # number of neighbors to consider
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # algorithm used to compute distances
        }
        # Perform grid search
        grid_search = GridSearchCV(model, params, cv=5, scoring='r2',n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        # Print and log the best parameters
        logging.info(f"Best parameters for KNeighborsRegressor: {best_params}")
        return best_model
    def predict(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred

    def evaluate(self, y_pred, y_test):
        """
        Evaluate the trained model using R-squared score.
        
        Parameters:
        ----------
        model : estimator
            Trained model.
        X_test : array-like
            Testing data.
        y_test : array-like
            Target variable for the testing data.
        
        Returns:
        -------
        score : float
            R-squared score of the model.
        """
        score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)

        # Print and log the evaluation score
        print("-------KNeighbors Regressor Model-------")
        logging.info(f"Model score: {score*100:.2f} -- Mean Squared Error: {mse:.2f} -- Mean Absolute Error: {mae:.2f} -- Max Error: {max_err:.2f}")
        return score, mse, mae, max_err


