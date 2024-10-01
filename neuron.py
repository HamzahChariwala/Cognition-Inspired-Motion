import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class NeuronResponseMap:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape

    def generate_dilation_map(self, radius):
        dilation_grid = np.zeros((self.rows, self.cols), dtype=float)

        # Get all indices where the grid has value 2
        indices = np.argwhere(self.grid == 2)

        for x, y in indices:
            dilation_grid[x, y] = 1  # Replace cells originally valued at 2 with 1

        for x in range(self.rows):
            for y in range(self.cols):
                if dilation_grid[x, y] != 1:  # Skip cells that are already set to 1
                    closest_distance = float('inf')
                    for (cx, cy) in indices:
                        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if distance <= radius and distance < closest_distance:
                            closest_distance = distance
                    if closest_distance <= radius:
                        # Normalize the distance to be between 0 and 1
                        normalized_distance = closest_distance / radius
                        dilation_value = np.sin((1 - normalized_distance) * (np.pi / 2))
                        dilation_grid[x, y] = max(dilation_grid[x, y], dilation_value)

        return dilation_grid

    def create_sine_circle(self, radius, center):
        cx, cy = center

        # Create a grid of distances from the center
        y, x = np.ogrid[:self.rows, :self.cols]
        distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Apply the sine function to get values between 0 and 1
        mask = distance_from_center <= radius
        circle_values = np.sin((np.pi / 2) * (1 - (distance_from_center / radius)))
        
        # Place the circle values in the grid, but ensure we don't go out of bounds
        self.grid[mask] = np.maximum(self.grid[mask], circle_values[mask])

        return self.grid

    def generate_tessellation_points(self, triangle_side_length):
        points = []
        height = np.sqrt(3) / 2 * triangle_side_length

        for i in range(-int(height), self.rows + int(height), int(height)):
            for j in range(-triangle_side_length, self.cols + triangle_side_length, triangle_side_length):
                # Offset every other row
                offset = (i // int(height)) % 2 * (triangle_side_length // 2)
                x = j + offset
                y = i

                points.append((x, y))
        
        return points
    
    def generate_boundary_buffer(self, radius):
        buffer_grid = np.zeros((self.rows, self.cols), dtype=float)

        # Get all indices where the grid has value 1 (boundaries)
        indices = np.argwhere(self.grid == 1)

        for x in range(self.rows):
            for y in range(self.cols):
                closest_distance = float('inf')
                for (bx, by) in indices:
                    distance = np.sqrt((x - bx) ** 2 + (y - by) ** 2)
                    if distance <= radius and distance < closest_distance:
                        closest_distance = distance
                if closest_distance <= radius:
                    # Normalize the distance to be between 0 and 1
                    normalized_distance = closest_distance / radius
                    buffer_value = np.sin((1 - normalized_distance) * (np.pi / 2))
                    buffer_grid[x, y] = buffer_value

        return buffer_grid

    def generate_landmark_circles(self, radius):
        # Create a new grid to store the result
        result_grid = np.zeros((self.rows, self.cols), dtype=float)

        # Get all indices where the grid has value 3 (landmarks)
        indices = np.argwhere(self.grid == 3)

        for center in indices:
            cx, cy = center
            # Create a grid of distances from the center
            x, y = np.ogrid[:self.rows, :self.cols]
            distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)

            # Apply the sine function to get values between 0 and 1
            mask = distance_from_center <= radius
            circle_values = np.sin((np.pi / 2) * (1 - (distance_from_center / radius)))

            # Place the circle values in the result grid
            result_grid[mask] = np.maximum(result_grid[mask], circle_values[mask])
            # Replace the landmark with 1
            result_grid[cx, cy] = 1

        return result_grid

    def export_grid(self):
        return self.grid

    def visualise_grid(self, grid):
        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='magma', origin='upper')
        plt.colorbar(ticks=[0, 0.5, 1], label='Grid Values')
        plt.clim(-0.1, 1.1)
        plt.title('Neuron Response Map')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()

# Example usage
# if __name__ == "__main__":

    # # Example grid from map.py
    # example_grid = np.array([
    #     [0, 0, 0, 0, 0],
    #     [0, 2, 2, 2, 0],
    #     [0, 0, 0, 2, 0],
    #     [0, 2, 0, 2, 0],
    #     [0, 0, 0, 0, 0]
    # ])

    # neuron_map = NeuronResponseMap(example_grid)
    # dilation_map = neuron_map.generate_dilation_map(radius=2)

    # print("Original Grid:")
    # print(example_grid)
    # print("\nDilation Map:")
    # print(dilation_map)

    # neuron_map.visualise_grid(dilation_map)

    # # Parameters
    # grid_size = (50, 50)
    # triangle_side_length = 15
    # radius = 5

    # # Create an empty grid
    # grid = np.zeros(grid_size)
    # functional_grid = NeuronResponseMap(grid)

    # # Generate tessellation points
    # tessellation_points = functional_grid.generate_tessellation_points(triangle_side_length)

    # # Apply sine circles to each tessellation point
    # for center in tessellation_points:
    #     grid_cell_map = functional_grid.create_sine_circle(radius, center)
    # functional_grid.visualise_grid(grid_cell_map)

if __name__ == "__main__":
    # Example grid from map.py
    example_grid = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 2],
        [1, 0, 2, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 2, 2]
    ])

    neuron_map = NeuronResponseMap(example_grid)
    boundary_buffer_map = neuron_map.generate_boundary_buffer(radius=2)

    print("Original Grid:")
    print(example_grid)
    print("\nBoundary Buffer Map:")
    print(boundary_buffer_map)

    neuron_map.visualise_grid(boundary_buffer_map)