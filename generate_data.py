import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt


def download_MNIST(class_nb= None):
    # Download MNIST dataset
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_data = mnist.data.numpy()
    mnist_labels = mnist.targets.numpy()
    # select only one instance per class
    mnist_data = mnist_data[np.unique(mnist_labels, return_index=True)[1]]
    if class_nb is not None:
        mnist_data = mnist_data[class_nb:class_nb+1]
    return mnist_data

def rotation_matrix_2d(angle):
    # matrix that acts on the indices of the pixels
    angle = np.deg2rad(angle)
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])

def get_rotated_index(batch, angle):
    # rotate a batch of images
    rotation_matrix = rotation_matrix_2d(angle)
    
    # Get indices
    (y, x) = np.indices(batch.shape[-2:])
    # But I want to make the rotation on the center of the image
    # So I need to put every vector in the center
    x = x - batch.shape[-1] / 2
    y = y - batch.shape[-2] / 2
    (h, w) = batch.shape[-2:]
    
    # Create coordinates for the image
    y, x = np.indices((h, w))
    coords = np.stack([x.ravel() - w / 2, y.ravel() - h / 2])
    
    # Apply the rotation matrix
    new_coords = rotation_matrix @ coords
    new_coords[0] += w / 2
    new_coords[1] += h / 2
    new_x, new_y = new_coords.round().astype(int)

    return (new_x, new_y), (x.ravel(), y.ravel())
    
def get_rotated_images(batch, angle):
    (new_x, new_y), (x, y) = get_rotated_index(batch, angle)
    mask = (new_x >= 0) & (new_x < batch.shape[-1]) & (new_y >= 0) & (new_y < batch.shape[-2])
    rotated_batch = torch.zeros_like(batch)
    rotated_batch[..., y[mask], x[mask]] = batch[..., new_y[mask], new_x[mask]]
    return rotated_batch

def visualize_batch(batch, nrow=8):
    grid = torchvision.utils.make_grid(batch, nrow=nrow)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def get_rotated_dataset(batch, n_samples):
    # sample n angles from 0 to 360
    angles = np.random.uniform(0, 360, n_samples)

    data_array = []
    for angle in angles:
        rotated_data = get_rotated_images(batch, angle)
        data_array.append(rotated_data)
    data_array = torch.stack(data_array)
    
    return data_array

def handle_data(batch, data_reference):
    # Normalize the data
    batch = batch.float() / 255
    # Reshape the data
    batch = batch.squeeze(2) # remove the channel dimension
    # flatten number dimension and angle dimension
    batch = batch.view(-1, 28*28)
    # standardize the data
    batch = (batch - batch.mean(dim=1, keepdim=True)) / batch.std(dim=1, keepdim=True)
    # randomly split train and test data 80/20
    n_train = int(0.8 * batch.shape[0])
    random_idx = torch.randperm(batch.shape[0])
    batch = batch[random_idx]
    train_data = batch[:n_train]
    test_data = batch[n_train:]
    
    data_reference = data_reference.float() / 255
    data_reference = data_reference.repeat(batch.shape[0] // data_reference.shape[0], 1)
    print(data_reference.shape)
    data_reference = data_reference[random_idx]
    reference_train = data_reference[:n_train]
    reference_test = data_reference[n_train:]
    # print shapes
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Reference train shape: {reference_train.shape}")
    print(f"Reference test shape: {reference_test.shape}")

    return train_data, test_data, reference_train, reference_test

def get_data(class_nb=4):
    # class_nb to only select one number,
    # in that way we can focus just on the rotation to 
    # understand what is going on!
    data = download_MNIST(class_nb=class_nb)
    data = torch.tensor(data).unsqueeze(1).float() # because MNIST is grayscale
    visualize_batch(data)
    
    data_array = get_rotated_dataset(data, 1000)
    train_data, test_data, reference_train, reference_test = handle_data(data_array, data.squeeze(1).reshape(10, -1)) # TMP be careful with this and number spec.

    return train_data, test_data, reference_train, reference_test


if __name__ == '__main__':
    get_data(class_nb=4)
    #data  = download_MNIST()
    #data = torch.tensor(data).unsqueeze(1).float() # because MNIST is grayscale
    ##rotated_data = get_rotated_images(data, 90)
    ##visualize_batch(rotated_data)
#
    #data_array = get_rotated_dataset(data, 6000)
    #train_data, test_data = handle_data(data_array)