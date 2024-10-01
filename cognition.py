import numpy as np
import matplotlib.pyplot as plt
from map import MapGeneration
from neuron import NeuronResponseMap


def create_neuron_maps(map):
    inflated_mask = NeuronResponseMap(map)
    dilation_map = inflated_mask.generate_dilation_map(6)

    boundary_cell = NeuronResponseMap(map)
    boundary_map = boundary_cell.generate_boundary_buffer(5)

    waypoints = NeuronResponseMap(map)
    checkpoint_map = waypoints.generate_landmark_circles(8)

    grid_size = map.shape
    triangle_side_length = 15
    radius = 5

    grid = np.zeros(grid_size)
    functional_grid = NeuronResponseMap(grid)
    tessellation_points = functional_grid.generate_tessellation_points(triangle_side_length)
    for center in tessellation_points:
        grid_cell_map = functional_grid.create_sine_circle(radius, center)

    return [dilation_map, boundary_map, checkpoint_map, grid_cell_map]


def plot_multiple_grids(grids, color_schemes):
    num_grids = len(grids)
    fig, axes = plt.subplots(1, num_grids, figsize=(3 * num_grids, 3))

    if num_grids == 1:
        axes = [axes]

    for i, (grid, color_scheme) in enumerate(zip(grids, color_schemes)):
        ax = axes[i]
        im = ax.imshow(grid, cmap=color_scheme, origin='upper')
        # ax.set_title(f'Grid {i + 1}')
        # ax.set_xlabel('Columns')
        # ax.set_ylabel('Rows')
        # fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1], label='Grid Values')

    plt.show()


if __name__ == "__main__":

    # Generate initial map
    sample1 = MapGeneration(50, 50)
    generated_grid = sample1.generate_map_1()

    # Apply dilation map
    inflated_mask = NeuronResponseMap(generated_grid)
    dilation_map = inflated_mask.generate_dilation_map(6)

    boundary_cell = NeuronResponseMap(generated_grid)
    boundary_map = boundary_cell.generate_boundary_buffer(5)

    waypoints = NeuronResponseMap(generated_grid)
    checkpoint_map = waypoints.generate_landmark_circles(8)

    # Generate tessellation points and sine circles
    grid_size = generated_grid.shape
    triangle_side_length = 15
    radius = 5

    grid = np.zeros(grid_size)
    functional_grid = NeuronResponseMap(grid)
    tessellation_points = functional_grid.generate_tessellation_points(triangle_side_length)
    for center in tessellation_points:
        grid_cell_map = functional_grid.create_sine_circle(radius, center)

    # Grids to be plotted and their corresponding color schemes
    grids = [generated_grid, dilation_map, boundary_map, checkpoint_map, grid_cell_map]
    color_schemes = ['gray', 'magma', 'magma', 'magma', 'magma']

    # Plot the multiple grids
    plot_multiple_grids(grids, color_schemes)

