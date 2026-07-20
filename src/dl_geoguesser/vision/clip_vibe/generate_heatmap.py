
import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_and_display_heatmap():
    # Generate a random 7x7 array for the heatmap
    heatmap_data = np.random.rand(7, 7)

    # Upscale the heatmap to 250x500 using bilinear interpolation
    upscaled_heatmap_data = cv2.resize(heatmap_data, (500, 250), interpolation=cv2.INTER_LINEAR)

    # Display the heatmap
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title('Upscaled Random 7x7 Heatmap (250x500)')
    plt.show()


if __name__ == "__main__":
    generate_and_display_heatmap()
