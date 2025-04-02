# For each file in ./states_density, we will create a gif that shows the density of the states.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib import rc
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from PIL import Image

def create_gif(dir_name: str):

    # Sort the files based on the number in the filename, ignoring invalid files
    files = sorted(
        [f for f in os.listdir(f"./{dir_name}") if f.split(".")[0].isdigit()],
        key=lambda x: int(x.split(".")[0])
    )

    # Create a gif by iterating over the files
    fig, ax = plt.subplots()
    ax.axis('off')  # Remove the axis
    ims = []
    for file in files:
        if file.endswith(".png"):  # Ensure only .png files are processed
            img = Image.open(f"./{dir_name}/{file}")
            data = np.array(img)  # Convert the image to a NumPy array
            im = ax.imshow(data, cmap='hot', animated=True)
            ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)  # Slower GIF with increased interval
    ani.save(f"./gifs/{dir_name}.gif")  # Save the animation as a single gif
    plt.close(fig)
    print("Created gif for all files")
    print("Done")

if __name__ == "__main__":
    dir_name = "states_density"
    create_gif(dir_name)
    dir_name = "nn_values"
    create_gif(dir_name)
    dir_name = "policy_values"
    create_gif(dir_name)
    dir_name = "variances"
    create_gif(dir_name)

