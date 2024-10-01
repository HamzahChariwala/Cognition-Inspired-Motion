import numpy as np
import glob
import random
import joblib

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.neural_network import MLPRegressor


file_paths = glob.glob('map1.v2.npy')
all_data = []
for file_path in file_paths:
    data = np.load(file_path, allow_pickle=True)
    all_data.extend(data)

random.shuffle(all_data)
combined_data = np.array(all_data)

print(combined_data.shape)

# # for i, data_point in enumerate(data):
# #     print(f"Data Point {i + 1}")
# #     print("Initial Position:", data_point['initial_position'])
# #     print("Final Position:", data_point['final_position'])
# #     print("Distance Traveled:", data_point['distance_traveled'])
# #     print("Angle of Approach:", data_point['angle_of_approach'])
# #     print("Gradient", data_point['gradient'])
# #     print("Angle of Deflection:", data_point['angle_of_deflection'])
# #     print("Initial Neurons:", data_point['initial_neurons'])
# #     print("Final Neurons:", data_point['final_neurons'])
# #     print("Reward 3x3 Array:", data_point['reward_3x3'])
# #     print() 

features = []
labels = []

for data_point in combined_data:
    initial_neurons = data_point['initial_neurons']
    final_neurons = data_point['final_neurons']
    distance_traveled = data_point['distance_traveled']
    angle_of_approach = data_point['angle_of_approach']
    # gradient = data_point['gradient']
    angle_of_deflection = data_point['angle_of_deflection']
    # reward_3x3 = np.array(data_point['reward_3x3']).flatten()  # Flatten the 3x3 array to 1D
    
    # Combine all features into a single list
    # feature_vector = list(initial_neurons) + list(final_neurons) + [distance_traveled, angle_of_approach, gradient] + list(reward_3x3)
    feature_vector = list(initial_neurons) + list(final_neurons) + [distance_traveled, angle_of_approach]
    features.append(feature_vector)
    
    # Use angle of deflection as the label
    labels.append(angle_of_deflection)

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Initialize and train the neural network
# model = MLPRegressor(hidden_layer_sizes=(100,), 
#                      max_iter=500, 
#                      activation='tanh', 
#                      solver='sgd', 
#                      learning_rate='constant')
# model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Mean Squared Error:", mse)
# print("R-squared:", r2)

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive'],
}

# Initialize the GridSearchCV with MLPRegressor and the parameter grid
grid_search = GridSearchCV(MLPRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best parameters found: ", best_params)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error of best model:", mse)
print("R-squared of best model:", r2)



# joblib.dump(model, 'trained_model.joblib')

# Load the model from the file
# model = joblib.load('trained_model.joblib')

# # Make predictions with the loaded model
# new_predictions = model.predict(X_test)
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)

# print("Mean Squared Error of best model:", mse)
# print("R-squared of best model:", r2)
# print("Mean Absolute Error of best model:", mae)
