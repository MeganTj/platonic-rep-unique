import random
import string
import torch
import numpy as np
import os
from PIL import Image

def text_perturbation_list(text_list, operations, percentage_perturbation):

    print(text_list[:10], operations, percentage_perturbation)

    if len(operations) == 0:
        return text_list
    
    operation = operations[0]

    if operation == "permute":
        random.shuffle(text_list)
        modified_list = text_list
    else:
        modified_list = [text_perturbatation(text, operation, percentage_perturbation) for text in text_list]
    
    return text_perturbation_list(modified_list, operations[1:], percentage_perturbation)


def text_perturbatation(text, operation, percentage_perturbation):

    if not (0 <= percentage_perturbation <= 100):
        raise ValueError("Percentage must be between 0 and 100.")

    num_to_modify = int(len(text) * percentage_perturbation / 100)

    if operation == "delete":
        deletion_set = set(random.sample(range(len(text)), num_to_modify))
        return "".join([c for i, c in enumerate(text) if i not in deletion_set])

    elif operation == "replace":
        replace_set = set(random.sample(range(len(text)), num_to_modify))
        return "".join([c if i not in replace_set else random.choice(string.ascii_letters) for i, c in enumerate(text)]) 
    

import torch

def image_perturbation_list(images, noise_percentage):
    """
    Replaces a specified percentage of pixels in each image with random noise based on
    the mean and variance of each individual image.
    
    Args:
        images (torch.Tensor): Batch of images (batch_size, channels, height, width).
        noise_percentage (float): Percentage of pixels to replace with noise (0 to 100).
    
    Returns:
        torch.Tensor: Images with a percentage of pixels replaced with random noise.
    """
    batch_size, channels, height, width = images.shape

    # Initialize tensor for noise
    noise = torch.zeros_like(images).cuda()
    
    # Generate noise for each image based on its mean and variance
    for idx in range(batch_size):
        img = images[idx]  # Extract individual image
        mean = img.mean().item()  # Compute mean of the image
        std = img.std().item()    # Compute standard deviation of the image
        
        # Generate noise with the computed mean and standard deviation
        noise[idx] = torch.normal(mean=mean, std=std, size=img.shape).cuda()
    
    # Calculate the threshold for the mask based on the noise percentage
    threshold = noise_percentage / 100.0

    # Generate binary masks to select pixels to be replaced
    masks = (torch.rand(batch_size, 1, height, width).cuda() < threshold).float()
    
    # Expand the mask to match the number of channels
    masks = masks.expand(-1, channels, -1, -1)
    
    # Replace selected pixels with noise
    images_with_noise = images * (1 - masks) + noise * masks
    
    return images_with_noise

def save_image(tensor, filename, normalize_range=True):
    """
    Save a tensor as an image file.
    
    Args:
        tensor (torch.Tensor): The image tensor to be saved. It should have shape (C, H, W).
        filename (str): The file path where the image will be saved.
        normalize_range (bool): Whether to normalize the tensor values to [0, 255] if it's not in that range.
    """
    # If the tensor is in range [0, 1], scale it to [0, 255]
    if normalize_range:
        tensor = tensor.clone()  # Avoid modifying the original tensor
        tensor = tensor * 255.0
        tensor = tensor.clamp(0, 255).to(torch.uint8)

    # Convert the tensor to a NumPy array
    image_np = tensor.cpu().numpy()
    
    # If the image is in (C, H, W), transpose to (H, W, C) for saving
    if image_np.ndim == 3:
        image_np = np.transpose(image_np, (1, 2, 0))

    # Create a PIL image from the NumPy array
    image_pil = Image.fromarray(image_np)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the image
    image_pil.save(filename)

    print(f"Image saved at {filename}")