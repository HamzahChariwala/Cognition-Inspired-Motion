import joblib
import numpy as np
import matplotlib.pyplot as plt
from map import MapGeneration
import cognition as Neuron
import movement as mv

# Load the model from the file
model = joblib.load('trained_model.joblib')

def visualise_grid(grid, visited_cells=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='magma', origin='upper')
    plt.colorbar(ticks=[0, 0.5, 1], label='Grid Values')
    plt.clim(-0.1, 1.1)
    plt.title('Neuron Response Map')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    if visited_cells:
        visited_cells = np.array(visited_cells)
        plt.scatter(visited_cells[:, 1], visited_cells[:, 0], c='blue', s=50, label='Visited Cells')
        plt.legend()

    plt.show()

sample_map = MapGeneration(50, 50)
validation_grid = sample_map.generate_map_1()
visualise_grid(validation_grid)

[obstacle, boundary, checkpoint, grid_cell] = Neuron.create_neuron_maps(validation_grid)
neuron_maps = {
    'grid_cell': grid_cell,
    'boundary': boundary,
    'obstacle': obstacle
}

def sample_neurons(neuron_maps, position):
    x, y = position
    grid_cell_value = neuron_maps['grid_cell'][x, y]
    boundary_value = neuron_maps['boundary'][x, y]
    obstacle_value = neuron_maps['obstacle'][x, y]
    return [grid_cell_value, boundary_value, obstacle_value]

def calculate_features(initial_neurons, final_neurons, start_position, current_position):
    distance_traveled = np.linalg.norm(np.array(current_position) - np.array(start_position))
    angle_of_approach = mv.calculate_angle(start_position, current_position)
    features = initial_neurons + final_neurons + [distance_traveled, angle_of_approach]
    return np.array(features).reshape(1, -1)

def take_step(grid, ML_model, position, neuron_maps, steps=5):
    current_position = position
    initial_neurons = sample_neurons(neuron_maps, current_position)
    neuron_data = [initial_neurons]
    trajectory = [current_position]

    for _ in range(steps):
        next_position = mv.random_step(current_position, grid)
        neuron_values = sample_neurons(neuron_maps, next_position)
        neuron_data.append(neuron_values)
        trajectory.append(next_position)
        current_position = next_position

    initial_neurons = neuron_data[0]
    final_neurons = neuron_data[-1]

    while True:
        feature_vector = calculate_features(initial_neurons, final_neurons, trajectory[-steps], current_position)
        
        # Predict the next move
        prediction = ML_model.predict(feature_vector)[0]
        print("\n")
        print(f"Prediction: {prediction}")
        
        # Convert prediction to next position
        print(f"Current: {current_position}")
        next_position = mv.calculate_next_position(current_position, trajectory[-steps], prediction)
        print(f"Next: {next_position}")
        
        # Ensure the next position is valid (not a 1 or 2 cell)
        while validation_grid[next_position] in [1, 2]:
            next_position = mv.random_step(current_position, grid)
        
        neuron_values = sample_neurons(neuron_maps, next_position)
        final_neurons = neuron_values  # Update final neurons with the most recent neuron values

        trajectory.append(next_position)
        current_position = next_position
        
        # Stop condition: Add your stopping condition here (e.g., a specific number of steps)
        if len(trajectory) > steps * 100:  # Example condition to stop after twice the initial steps
            break

    return trajectory

[row, col] = mv.start_position(validation_grid)

trajectory = take_step(validation_grid, model, (row, col), neuron_maps, steps=5)
visualise_grid(validation_grid, visited_cells=trajectory)

