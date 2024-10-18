import tifffile as tiff
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        # Weight matrix of size (hidden_units, visible_units)
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.1)
        # Bias for visible units
        self.b_v = nn.Parameter(torch.zeros(visible_units))
        # Bias for hidden units
        self.b_h = nn.Parameter(torch.zeros(hidden_units))

    def sample_hidden(self, v):
        # Compute hidden probabilities and sample activations
        h_prob = torch.sigmoid(F.linear(v, self.W, self.b_h))
        return h_prob, torch.bernoulli(h_prob)

    def sample_visible(self, h):
        # Compute visible probabilities and sample activations
        v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.b_v))
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, v):
        # One step of Gibbs sampling (reconstruction of visible units)
        _, h = self.sample_hidden(v)
        v_recon_prob, _ = self.sample_visible(h)
        return v_recon_prob

    def contrastive_divergence(self, v, lr=0.01):
        # Positive phase
        h_prob, h_sample = self.sample_hidden(v)
        # Negative phase
        v_recon_prob, _ = self.sample_visible(h_sample)
        h_recon_prob, _ = self.sample_hidden(v_recon_prob)

        # Compute the positive and negative gradients
        positive_grad = torch.matmul(h_prob.t(), v)
        negative_grad = torch.matmul(h_recon_prob.t(), v_recon_prob)

        # Update weights and biases
        self.W.data += lr * (positive_grad - negative_grad) / v.size(0)
        self.b_v.data += lr * torch.sum(v - v_recon_prob, dim=0) / v.size(0)
        self.b_h.data += lr * torch.sum(h_prob - h_recon_prob, dim=0) / v.size(0)


# Load the TIF file
tif_data = tiff.imread('modified_dataset.tif')  # Replace 'your_file.tif' with your actual file name

# Check the shape (should be 2950, 815, 224)
print("TIF Data Shape:", tif_data.shape)

# Reshape the data into a 2D matrix where each row is a pixel and each column is a spectral band
flattened_data = tif_data.reshape(-1, tif_data.shape[-1])  # Shape: (2950 * 815, 224)

# Convert to a PyTorch tensor
data_tensor = torch.tensor(flattened_data, dtype=torch.float32)
print("Flattened Data Shape:", data_tensor.shape)  # Should be (2404250, 224)

visible_units = 224  # Input features (spectral bands)
hidden_units_1 = 128  # First hidden layer size

# Initialize the first RBM
rbm1 = RBM(visible_units, hidden_units_1)

# Train the first RBM
epochs = 5
batch_size = 100  # Use batches for large data
for epoch in range(epochs):
    loss = 0
    for i in range(0, data_tensor.size(0), batch_size):
        batch = data_tensor[i:i+batch_size]
        v_recon = rbm1(batch)
        rbm1.contrastive_divergence(batch)
        loss += torch.mean((batch - v_recon) ** 2)  # Mean squared error loss
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


hidden_units_2 = 32  # Second hidden layer size

# Initialize the second RBM
rbm2 = RBM(hidden_units_1, hidden_units_2)

# Get output from the first RBM
hidden_layer_1_output = []
for i in range(0, data_tensor.size(0), batch_size):
    batch = data_tensor[i:i+batch_size]
    _, h_sample = rbm1.sample_hidden(batch)
    hidden_layer_1_output.append(h_sample)
hidden_layer_1_output = torch.cat(hidden_layer_1_output, dim=0)

# Train the second RBM
for epoch in range(epochs):
    loss = 0
    for i in range(0, hidden_layer_1_output.size(0), batch_size):
        batch = hidden_layer_1_output[i:i+batch_size]
        v_recon = rbm2(batch)
        rbm2.contrastive_divergence(batch)
        loss += torch.mean((batch - v_recon) ** 2)  # Mean squared error loss
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


class DBN(nn.Module):
    def __init__(self, rbm1, rbm2):
        super(DBN, self).__init__()
        self.rbm1 = rbm1
        self.rbm2 = rbm2

    def forward(self, x):
        _, h1 = self.rbm1.sample_hidden(x)
        _, h2 = self.rbm2.sample_hidden(h1)
        return h2  # Final abstract features

# Initialize the DBN
dbn = DBN(rbm1, rbm2)

# Test the DBN with the original data
abstract_features = []
for i in range(0, data_tensor.size(0), batch_size):
    batch = data_tensor[i:i+batch_size]
    features = dbn(batch)
    abstract_features.append(features)
abstract_features = torch.cat(abstract_features, dim=0)

# Assuming 'abstract_features' has the shape (2407250, 64)
print("Abstract Features Shape:", abstract_features.shape)  # Should be (2407250, 64)

# Reshape back to the original spatial dimensions: (2950, 815, 64)
reshaped_features = abstract_features.reshape(2950, 815, -1)
print("Reshaped Features Shape:", reshaped_features.shape)  # Should be (2950, 815, 64)

# Convert to NumPy array (if necessary) before saving as TIF
reshaped_features_np = reshaped_features.detach().cpu().numpy()

# Save the reshaped abstract features as a new TIF file
tiff.imwrite('dbn_output.tif', reshaped_features_np)

print("New TIF file 'dbn_output.tif' created successfully!")