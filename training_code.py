import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# load data

train_data = np.load('../data/ptrain.npy')
test_data = np.load('../data/ptest.npy')
targets = pd.read_csv('../data/training_data_targets.csv', header=None).to_numpy()[:,0]

#log scale targets
targets = np.log(targets)
outfile_results = open('results/scores/scores_r2_no_pca.txt', 'w')

#start with linear regression
#define the model
model = LinearRegression()
# define the grid search
grid = dict()
grid['fit_intercept'] = [True, False]
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
# perform the search
grid_result = grid_search.fit(train_data, targets)
# summarize the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
outfile_results.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
# save best parameters as a dictionary
best_params = grid_result.best_params_
# save as a json file

with open('results/linear_regression_best_params_nmse_no_pca.json', 'w') as f:
    json.dump(best_params, f)
# save the model
joblib.dump(grid_result.best_estimator_, 'results/linear_regression_nmse_no_pca.pkl')
# evaluate the model
y_pred = grid_result.predict(test_data)
#save results
np.save('results/linear_regression_results_nmse_no_pca.npy', y_pred)

# ridge regression
# define the model
model = Ridge()
# define the grid search
grid = dict()
grid['alpha'] = [0.1, 1, 10]
grid['max_iter'] = [1000, 2000, 4000]
grid['solver'] = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
# perform the search
grid_result = grid_search.fit(train_data, targets)
# summarize the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
outfile_results.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
best_params = grid_result.best_params_
# save as a json file

with open('results/ridge_regression_best_params_nmse_no_pca.json', 'w') as f:
    json.dump(best_params, f)
# save the model
joblib.dump(grid_result.best_estimator_, 'results/ridge_regression_nmse_no_pca.pkl')
# evaluate the model
y_pred = grid_result.predict(test_data)
#save results
np.save('results/ridge_regression_results_nmse_no_pca.npy', y_pred)

# decision tree regressor
# define the model
model = DecisionTreeRegressor()
# define the grid search
grid = dict()
grid['max_depth'] = [None, 10, 100, 500]
grid['min_samples_split'] = [2, 4, 8]
grid['min_samples_leaf'] = [1, 2, 4]
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
# perform the search

grid_result = grid_search.fit(train_data, targets)
# summarize the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
outfile_results.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
best_params = grid_result.best_params_
# save as a json file

with open('results/decision_tree_regressor_best_params_nmse_no_pca.json', 'w') as f:
    json.dump(best_params, f)
# save the model
joblib.dump(grid_result.best_estimator_, 'results/decision_tree_regressor_nmse_no_pca.pkl')
# evaluate the model
y_pred = grid_result.predict(test_data)
#save results
np.save('results/decision_tree_regressor_results_nmse_no_pca.npy', y_pred)

# ada boost regressor
# define the model
model = AdaBoostRegressor()
# define the grid search
grid = dict()
grid['n_estimators'] = [10, 100, 1000]
grid['learning_rate'] = [0.001, 0.01, 0.1]
grid['loss'] = ['linear', 'square', 'exponential']
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='r2')
#perform the search

grid_result = grid_search.fit(train_data, targets)
# summarize the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
outfile_results.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
best_params = grid_result.best_params_
# save as a json file

with open('results/ada_boost_regressor_best_params_r2_no_pca.json', 'w') as f:
    json.dump(best_params, f)
# save the model
joblib.dump(grid_result.best_estimator_, 'results/ada_boost_regressor_r2_no_pca.pkl')
# evaluate the model
y_pred = grid_result.predict(test_data)
#save results
np.save('results/ada_boost_regressor_results_r2_no_pca.npy', y_pred)

# random forest regressor
# define the model
model = RandomForestRegressor()
# define the grid search
grid = dict()
grid['n_estimators'] = [10, 100, 1000]
grid['max_depth'] = [ 5, 10, 50]
grid['min_samples_split'] = [2, 4, 8]
grid['min_samples_leaf'] = [1, 2, 4]
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='r2')
# perform the search

grid_result = grid_search.fit(train_data, targets)
# summarize the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
outfile_results.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
best_params = grid_result.best_params_
# save as a json file

with open('results/random_forest_regressor_best_params_r2_no_pca.json', 'w') as f:
    json.dump(best_params, f)
# save the model
joblib.dump(grid_result.best_estimator_, 'results/random_forest_regressor_r2_no_pca.pkl')
# evaluate the model
y_pred = grid_result.predict(test_data)
#save results
np.save('results/random_forest_regressor_results_r2_no_pca.npy', y_pred)

# xgboost regressor
# define the model
model = XGBRegressor()
# define the grid search
grid = dict()
grid['n_estimators'] = [10, 100, 1000]
grid['max_depth'] = [ 5,10]
grid['learning_rate'] = [0.1,0.01,0.001]
grid['booster'] = ['gbtree']

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=5, scoring='r2')
# perform the search

grid_result = grid_search.fit(train_data, targets)
# summarize the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
outfile_results.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
best_params = grid_result.best_params_
# save as a json file

with open('results/xgboost_regression_best_params_r2_no_pca.json', 'w') as f:
    json.dump(best_params, f)
# save the model
joblib.dump(grid_result.best_estimator_, 'results/xgboost_regressor_r2_no_pca.pkl')
# evaluate the model
y_pred = grid_result.predict(test_data)
#save results
np.save('results/xgboost_regressor_results_r2_no_pca.npy', y_pred)