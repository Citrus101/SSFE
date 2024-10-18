import tifffile as tiff
import numpy as np
import pandas as pd

# Open the TIFF file
tif_path = 'dataset.tif'
image = tiff.imread(tif_path)

m, n, L = image.shape  # Get the dimensions of the image

# Initialize a list to hold all absolute difference vectors
all_abs_diff_vectors = []

# Function to get absolute difference vectors for a specific pixel (x, y)
def get_vectors(image, x, y):
    x_min, x_max = max(0, x - 1), min(m, x + 2)
    y_min, y_max = max(0, y - 1), min(n, y + 2)

    y_vector = image[x, y, :]  # Get the vector at the center pixel
    abs_diff_vectors = []

    # Loop through the neighboring pixels (3x3 window)
    count = 0
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            if i == x and j == y:  # Exclude the center pixel
                continue  # Skip this iteration
            Si = image[i, j, :]  # Get the neighboring vector
            print(count);
            count += 1;
            abs_diff = np.abs(y_vector - Si)  # Calculate absolute difference
            abs_diff_vectors.append(abs_diff)

    return abs_diff_vectors
# Example usage
x, y = 100, 100  # Center pixel coordinates
abs_diff_vectors = get_vectors(image, x, y)

# Convert the list of vectors to a DataFrame for easy saving
abs_diff_df = pd.DataFrame(abs_diff_vectors)

# Save to CSV file
output_path = 'abs_diff_vectors.csv'
abs_diff_df.to_csv(output_path, index=False)