import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 2))

# Use seaborn's color_palette to create a colormap
cmap = sns.color_palette("RdYlBu_r", as_cmap=True)

# Create a dummy image with a gradient from 0 to 1
gradient_image = np.linspace(0, 1, 256).reshape(1, -1)

# Plot the horizontal color bar
im = ax.imshow(gradient_image, cmap=cmap, aspect="auto", extent=[0, 1, 0, 1])

# Set axis ticks and labels
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(["0", "0.5", "1"])

# Hide the y-axis
ax.yaxis.set_visible(False)

# Show the color bar
plt.show()
