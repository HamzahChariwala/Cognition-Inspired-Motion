import numpy as np
import random
from map import MapGeneration
import reinforcement as RL
import movement as mv
import cognition as Neuron

def sample_neurons(neuron_maps, position):
    x, y = position
    grid_cell_value = neuron_maps['grid_cell'][x, y]
    boundary_value = neuron_maps['boundary'][x, y]
    obstacle_value = neuron_maps['obstacle'][x, y]
    return grid_cell_value, boundary_value, obstacle_value

def collect_data(map1, neuron_maps, reward_array, iterations=10):
    data = []

    for _ in range(iterations):
        # Generate a random starting position (not on a 1 or 2)
        start_position = (random.randint(0, 49), random.randint(0, 49))
        while map1[start_position] in [1, 2]:
            start_position = (random.randint(0, 49), random.randint(0, 49))

        initial_neurons = sample_neurons(neuron_maps, start_position)

        current_position = start_position
        trajectory = []

        for step in range(5):
            next_position = mv.select_next_position(current_position, reward_array, map1)
            while map1[next_position] in [1, 2]:
                next_position = mv.select_next_position(current_position, reward_array, map1)

            trajectory.append(next_position)
            current_position = next_position

        final_neurons = sample_neurons(neuron_maps, current_position)
        distance_traveled = np.linalg.norm(np.array(current_position) - np.array(start_position))
        angle_of_approach = mv.calculate_angle(start_position, current_position)
        if angle_of_approach < 0:
            angle_of_approach += 360

        gradient = mv.calculate_gradient(start_position, current_position)

        surrounding_positions = mv.get_surrounding_positions(current_position)
        best_position = mv.select_best_position(surrounding_positions, reward_array, map1)
        angle_of_deflection = mv.calculate_angle(current_position, best_position) - angle_of_approach
        if angle_of_deflection < 0:
            angle_of_deflection += 360

        reward_3x3 = []
        for dx in range(-1, 2):
            row = []
            for dy in range(-1, 2):
                pos = (current_position[0] + dx, current_position[1] + dy)
                if 0 <= pos[0] < map1.shape[0] and 0 <= pos[1] < map1.shape[1]:
                    row.append(reward_array[pos[0], pos[1]])
                else:
                    row.append(None)
            reward_3x3.append(row)

        data.append({
            'initial_position': start_position,
            'final_position': current_position,
            'initial_neurons': initial_neurons,
            'final_neurons': final_neurons,
            'distance_traveled': distance_traveled,
            'angle_of_approach': angle_of_approach,
            'gradient': gradient,
            'angle_of_deflection': angle_of_deflection,
            'reward_3x3': reward_3x3
        })

    return data

# Initialize the map
sample1 = MapGeneration(50, 50)
map1 = sample1.generate_map_1()

# Generate neuron maps
obstacle_map, boundary_map, checkpoint_map, grid_cell_map = Neuron.create_neuron_maps(map1)
neuron_maps = {
    'grid_cell': grid_cell_map,
    'boundary': boundary_map,
    'obstacle': obstacle_map
}

# Create reward array using RL approach
reward_array = RL.average_array(map1, 15)

# Collect data
data = collect_data(map1, neuron_maps, reward_array, iterations=10000)

# Store data for training ML model
np.save('map1.v2.npy', data)
