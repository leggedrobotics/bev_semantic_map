import cv2
import numpy as np

def create_black_image(width, height, filename):
    # Create a black image with the specified width and height
    black_image = 255 * np.ones((height, width, 3), dtype=np.uint8)
    black_image[:, :, :] = 0

    # Save the image to the specified filename
    cv2.imwrite(filename, black_image)
    print(f"Black image of size {width}x{height} saved as {filename}")


if __name__ == "__main__":
    width = 512
    height = 512

    create_black_image(width, height, "/home/rschmid/img.jpg")
