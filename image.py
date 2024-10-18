import tifffile as tiff
import matplotlib.pyplot as plt

# Read the TIF file
image = tiff.imread('dataset.tif')

# Display the shape of the image
print(image.shape)

int SPECTRAL_BAND = 115

plt.figure(figsize=(8, 6))
plt.imshow(image[:, :, SPECTRAL_BAND] )
plt.axis('off')  # Hide axis labels
plt.show()
