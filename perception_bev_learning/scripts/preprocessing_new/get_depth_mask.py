import numpy as np

image_path = "/home/jonfrey/Downloads/back_mask_396_640.png"

import imageio

# Read the image
image = imageio.imread(image_path)
print(image)
# Create a binary mask
binary_mask = image == 0  # Change 128 to whatever threshold you want

# If you want to save the binary mask
imageio.imsave("binary_mask.png", binary_mask.astype(np.uint8) * 255)
