import tifffile as tiff
import numpy as np
import pandas as pd

def compute_new_vector(tif_path):
    # Load the TIFF image
    image = tiff.imread(tif_path)
    
    # Get dimensions of the image
    m, n, L = image.shape  

    # Initialize a list to hold the new vectors y'
    new_vectors = []

    # Function to calculate the alpha values and new vector y'
    def calculate_y_prime(x, y):
        # Define the 3x3 neighborhood limits
        x_min, x_max = max(0, x - 1), min(m, x + 2)
        y_min, y_max = max(0, y - 1), min(n, y + 2)

        # Get the center pixel vector y
        y_vector = image[x, y, :]  
        abs_diff_vectors = []
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
                abs_diff_vectors.append(d_i)

                # Calculate the alpha value
                alpha_i = 1 - np.exp(-10 * d_i)
                alpha_values.append(alpha_i)

        # Calculate the new vector y'
        sum_alpha = 8  # Given assumption
        y_prime = np.sum(np.array(alpha_values), axis=0) / sum_alpha

        return y_prime

    # Loop through all pixels in the image
    for x in range(m):
        for y in range(n):
            y_prime = calculate_y_prime(x, y)
            new_vectors.append(y_prime)

    # Convert the list of new vectors to a DataFrame for easy saving
    new_vectors_df = pd.DataFrame(new_vectors)

    # Save the new vectors to a CSV file
    output_path = 'new_vectors.csv'
    new_vectors_df.to_csv(output_path, index=False)

    print(f"New vectors y' saved to {output_path}.")

# Example usage
compute_new_vector('dataset.tif')
