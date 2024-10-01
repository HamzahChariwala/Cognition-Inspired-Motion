import numpy as np
import matplotlib.pyplot as plt

class MapGeneration:
    def __init__(self, rows, cols):
        self.grid = np.zeros((rows, cols), dtype=int)
        self.rows = rows
        self.cols = cols
        self.place_boundaries()
    
    def place_boundaries(self):
        self.grid[:3, :] = 1
        self.grid[-3:, :] = 1
        self.grid[:, :3] = 1
        self.grid[:, -3:] = 1

    def add_rectangle(self, top_left, bottom_right, value):
        x1, y1 = top_left
        x2, y2 = bottom_right
        self.grid[x1:x2+1, y1:y2+1] = value

    def add_rotated_rectangle(self, center, width, height, angle, value):
        cx, cy = center
        angle_rad = np.radians(angle)
        
        # Calculate half dimensions
        half_width = width / 2
        half_height = height / 2
        
        # Define the corners of the rectangle before rotation
        corners = np.array([
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height]
        ])
        
        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # Rotate the corners
        rotated_corners = np.dot(corners, rotation_matrix)
        
        # Translate corners back to the center
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy
        
        # Determine the bounding box of the rotated rectangle
        min_x = max(0, int(np.floor(rotated_corners[:, 0].min())))
        max_x = min(self.rows - 1, int(np.ceil(rotated_corners[:, 0].max())))
        min_y = max(0, int(np.floor(rotated_corners[:, 1].min())))
        max_y = min(self.cols - 1, int(np.ceil(rotated_corners[:, 1].max())))
        
        # Fill in the grid within the bounding box
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if self.point_in_polygon(x, y, rotated_corners):
                    self.grid[x, y] = value

    def point_in_polygon(self, x, y, polygon):
        # Ray casting algorithm for point in polygon
        num_vertices = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(num_vertices + 1):
            p2x, p2y = polygon[i % num_vertices]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def add_circle(self, center, radius, value):
        cx, cy = center
        for x in range(self.rows):
            for y in range(self.cols):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self.grid[x, y] = value

    def add_checkpoint(self, position, value):
        x, y = position
        if 0 <= x < self.rows and 0 <= y < self.cols:
            self.grid[x, y] = value

    def export_as_array(self):
        return self.grid

    def display_grid(self):
        for row in self.grid:
            print(' '.join(map(str, row)))

    def visualise_grid(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='gray', origin='upper')
        plt.colorbar(ticks=[0, 1, 2], label='Grid Values')
        plt.clim(-0.5, 2.5)
        plt.title('Grid Visualization')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()

    def generate_map_1(self):
        self.place_boundaries()
        self.add_rectangle((19, 34), (49, 49), 1) 
        self.add_rectangle((5, 30), (14, 44), 2)
        self.add_rectangle((30, 5), (44, 11), 2) 
        self.add_rectangle((25, 5), (30, 20), 2) 
        self.add_rotated_rectangle((8, 9), 6, 10, 35, 2) 
        self.add_rectangle((25, 31), (43, 33), 2)
        self.add_rectangle((34, 16), (42, 20), 2)
        self.add_circle((15, 7), 2, 2)
        self.add_checkpoint((45, 18), 3) 
        self.add_checkpoint((5, 27), 3) 
        self.add_checkpoint((19, 10), 3)
        return self.grid
    
    def generate_test_map(self):
        self.place_boundaries()
        self.add_rectangle((16, 24), (33, 35), 2) 
        self.add_rectangle((25, 26), (43, 33), 2)
        return self.grid

# Example usage
if __name__ == "__main__":
    sample_map = MapGeneration(50, 50)  # Grid size 50x50

    # Add boundary walls
    sample_map.place_boundaries()
    
    # Partitions
    # sample_map.add_rectangle((19, 34), (49, 49), 1) 

    # Dining table
    sample_map.add_rectangle((5, 30), (14, 44), 2)
    
    # Sofas
    sample_map.add_rectangle((30, 5), (44, 11), 2)  # Large Sofa
    sample_map.add_rectangle((25, 5), (30, 20), 2)  # Top-left sofa
    sample_map.add_rotated_rectangle((8, 9), 6, 10, 35, 2)  # Skewed Sky sofa
    
    # Console unit
    sample_map.add_rectangle((25, 31), (43, 33), 2)

    # Coffee tables
    sample_map.add_rectangle((34, 16), (42, 20), 2) # Big table
    sample_map.add_circle((15, 7), 2, 2) # Side table

    # Checkpoints
    sample_map.add_checkpoint((45, 18), 3) # By large coffee table
    sample_map.add_checkpoint((5, 27), 3) # By dining table
    sample_map.add_checkpoint((19, 10), 3) # By side table

    sample_map.display_grid()
    sample_map.visualise_grid()
