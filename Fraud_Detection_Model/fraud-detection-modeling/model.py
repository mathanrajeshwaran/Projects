from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris
import numpy as np 
import pandas as pd 



n_estimators_range = np.arange(20, 100, 20)

max_depth_range = np.arange(5, 30, 5)


param_grid = {
    'n_estimators': n_estimators_range,
    'max_depth': max_depth_range,

}

rf_classifier = RandomForestClassifier(random_state=64)


X = pd.read_csv("D:/Personal Documents/MATHAN PERSONAL DOCUMENTS/Technical Training/Data Science Projects/katacoda-scenarios/fraud-detection-data-prep/assets/features.csv")

y = pd.read_csv("D:/Personal Documents/MATHAN PERSONAL DOCUMENTS/Technical Training/Data Science Projects/katacoda-scenarios/fraud-detection-data-prep/assets/targets.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=128, test_size = 0.2)

print(X_train.head())
print(y_train.head())


print(X_test.head())
print(y_test.head())

print("DATA HAS BEEN SPLIT FOR TRAINING AND TESTING")
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='precision')

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_

y_pred = best_rf_model.predict(X_test)

print("OPTIMAL MODEL HAS BEEN DEFINED AND PREDICTIONS WERE MADE ON THE TEST SET")

import pickle

validation_data = X_test
validation_data['actual'] = y_test
validation_data['predicted'] = y_pred

validation_data.to_csv("validation_data.csv", index= False)

model_filename = 'random_forest_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(best_rf_model, model_file)

print(f"Random Forest model saved to {model_filename}")