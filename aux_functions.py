import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata




def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b




def build_pitch_grid(x_dim, y_dim, printGrid = False, num_random_points=100, min_fq=20, max_fq=2000, corner_pitch_values=None):
    """
    num_random_points -> number of random points. the rest is going to be interpolated
    corner_pitch_values -> corner pitch values so that the whole grid has values and it's no constant outide the last random points

    """
    try:
        # Define grid dimensions
        #x_dim, y_dim = 640, 480

        # Create a grid
        x = np.arange(x_dim)
        y = np.arange(y_dim)
        xx, yy = np.meshgrid(x, y)

        # Generate random points
        num_points = min(x_dim*y_dim, num_random_points)  # Number of random points
        random_x = np.random.randint(0, x_dim, size=num_points)
        random_y = np.random.randint(0, y_dim, size=num_points)
        random_points = np.array(list(zip(random_x, random_y)))

        # Assign random pitch values to random points
        random_pitch_values = np.random.uniform(min_fq, max_fq, size=num_points)  # MIDI or Hz

        # Define corner points
        corners = [(0, 0), (x_dim - 1, 0), (0, y_dim - 1), (x_dim - 1, y_dim - 1)]
        corner_pitch_values = [(min_fq+max_fq)/2]*4 if corner_pitch_values is None else corner_pitch_values #corner_pitch_values are gonna be the mid freq value if not initialized
        # Combine random points and corner points
        all_points = np.vstack((random_points, corners))
        all_pitch_values = np.concatenate((random_pitch_values, corner_pitch_values))

        # Interpolate across the grid
        interpolated_pitch = griddata(
            all_points, all_pitch_values, (xx, yy), method='cubic', fill_value=np.nan
        )

        interpolated_pitch = rescale_linear(interpolated_pitch, 200, 800) #hearing is between 20 and  20000

        if printGrid:
            # Visualization
            plt.figure(figsize=(10, 6))
            plt.imshow(interpolated_pitch, origin="lower", extent=(0, x_dim, 0, y_dim), cmap="viridis")
            plt.colorbar(label="Pitch Value")
            plt.scatter(*zip(*all_points), c=all_pitch_values, edgecolor="white", label="Defined Points")
            plt.legend()
            plt.title("2D Pitch Mapping with Corners Included")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

        return interpolated_pitch
    
    except Exception as e:
        print(f'Exception in build_pitch_grid: {e}')
