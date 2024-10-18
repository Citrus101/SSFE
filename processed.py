import tifffile as tiff
import numpy as np

# Load the TIFF image
input_path = 'dataset.tif'
image = tiff.imread(input_path)

# Check the shape of the image
print(f"Original image shape: {image.shape}")

# Create a mask for non-black pixels (assuming black is represented by 0s)
non_black_mask = np.any(image > 0, axis=-1)

# Get the indices of non-black pixels
rows, cols = np.nonzero(non_black_mask)

# Check if there are any non-black pixels
if rows.size == 0 or cols.size == 0:
    print("No non-black pixels found. Saving empty image.")
    processed_image = np.zeros_like(image)  # Create an empty image if all pixels are black
else:
    # Define the bounding box for non-black pixels
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    # Crop the image to the bounding box of non-black pixels
    processed_image = image[min_row:max_row + 1, min_col:max_col + 1]

# Save the processed image to a new TIFF file
output_path = 'processed.tif'
tiff.imwrite(output_path, processed_image)

print(f"Processed image saved as {output_path}.")
