import tifffile as tiff
import numpy as np

def compute_new_file(tif_path, output_path):
    # Load the TIFF image
    image = tiff.imread(tif_path)
    
    # Get dimensions of the image
    m, n, L = image.shape  

    # Initialize a new array to hold the new vectors y'
    new_vectors = np.zeros((m, n, L))

    # Function to calculate the alpha values and new vector y'
    def calculate_y_prime(x, y):
        # Define the 3x3 neighborhood limits
        x_min, x_max = max(0, x - 1), min(m, x + 2)
        y_min, y_max = max(0, y - 1), min(n, y + 2)

        # Get the center pixel vector y
        y_vector = image[x, y, :]  
        alpha_values = []

        # Loop through the neighboring pixels (3x3 window)
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                if i == x and j == y:  # Skip the center pixel
                    continue
                
                # Get the neighboring vector Si
                Si = image[i, j, :]
                # Calculate the absolute difference
                d_i = np.abs(y_vector - Si)

                # Calculate the alpha value
                alpha_i = 1 - np.exp(-10 * d_i)
                alpha_values.append(alpha_i)

        # Calculate the new vector y' using the given sum of alpha
        sum_alpha = 8  # Given assumption
        y_prime = np.sum(alpha_values, axis=0) / sum_alpha

        return y_prime

    # Loop through all pixels in the image
    for x in range(m):
        for y in range(n):
            new_vectors[x, y, :] = calculate_y_prime(x, y)

    # Save the new vectors as a TIFF file
    tiff.imwrite(output_path, new_vectors.astype(np.float32))  # Use appropriate dtype

    print(f"New vectors y' saved to {output_path}.")

# Example usage
compute_new_file'dataset.tif', 'modified_dataset.tif')
