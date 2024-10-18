import numpy as np
import tifffile as tiff

# Step 1: Load the TIF file
# Assuming the TIF file is named 'h_i_2.tif'
h_i_2 = tiff.imread('dbn_output.tif')  # Shape: (2950, 815, 32)

# Reshape to (m*n, C), where m*n is the number of pixels and C is the number of features
m, n, C = h_i_2.shape
h_i_2_reshaped = h_i_2.reshape(-1, C)  # Shape: (2407250, 32)

# Step 2: Compute the mean vector and covariance matrix
mu = np.mean(h_i_2_reshaped, axis=0)  # Mean vector shape: (32,)
Gamma = np.cov(h_i_2_reshaped, rowvar=False)  # Covariance matrix shape: (32, 32)

# Step 3: Calculate the inverse of the covariance matrix
Gamma_inv = np.linalg.inv(Gamma)

# Step 4: Compute spectral distance for each vector
D_spectral = np.zeros(h_i_2_reshaped.shape[0])  # Initialize array for distances

for i in range(h_i_2_reshaped.shape[0]):
    diff = h_i_2_reshaped[i] - mu  # Difference from mean
    D_spectral[i] = diff.T @ Gamma_inv @ diff  # Calculate D_spectral

# Reshape D_spectral back to original image dimensions (2950, 815)
D_spectral_reshaped = D_spectral.reshape(m, n)

# Optionally, save the D_spectral result as a new TIF file
tiff.imwrite('D_spectral.tif', D_spectral_reshaped)

print("Spectral distance calculated and saved as 'D_spectral.tif'")
