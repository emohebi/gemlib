import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def CustomCmap(from_rgb, to_rgb, name="custom_cmap"):

    # from color r,g,b
    r1,g1,b1 = from_rgb

    # to color r,g,b
    r2,g2,b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                     (1, r2, r2)),
             'green': ((0, g1, g1),
                       (1, g2, g2)),
             'blue': ((0, b1, b1),
                      (1, b2, b2))}

    cmap = LinearSegmentedColormap(name, cdict)
    return cmap


def setup():
    plt.style.use('ggplot')

    plt.rc("lines",
           linewidth=2.0,
           linestyle="-",
           color="black",
           marker="None",
           markeredgewidth=0.0,
           markersize=6.0,
           solid_joinstyle="round",
           solid_capstyle="round",
           antialiased=True)

    plt.rc("patch", linewidth=1.0, edgecolor="none", antialiased=True)

    plt.rc("xtick.major", size=4, width=1, pad=3)
    plt.rc("xtick.minor", size=2, width=1)
    plt.rc("xtick", direction="in")

    plt.rc("ytick.major", size=4, width=1, pad=3)
    plt.rc("ytick.minor", size=2, width=1)
    plt.rc("ytick", direction="in")

    plt.rc("savefig", pad_inches=0.25)
    plt.rc("grid", color="0.7", linestyle="solid", alpha="0.4")
    plt.rc('axes', labelpad=10, facecolor="white", edgecolor="black", labelcolor="black")

    cdict1 = {
        'red': ((0.0, 43.0 / 255.0, 43.0 / 255.0),
                (0.5, 171.0 / 255.0, 171.0 / 255.0),
                (1.0, 243.0 / 255.0, 243.0 / 255.0)),

        'green': ((0.0, 38.0 / 255.0, 38.0 / 255.0),
                  (0.5, 84.0 / 255.0, 84.0 / 255.0),
                  (1.0, 180.0 / 255.0, 180.0 / 255.0)),

        'blue': ((0.0, 94.0 / 255.0, 94.0 / 255.0),
                 (0.5, 137.0 / 255.0, 137.0 / 255.0),
                 (1.0, 79.0 / 255.0, 79.0 / 255.0))
    }
    fovio_cmap = LinearSegmentedColormap('Fovio', cdict1)
    plt.register_cmap(cmap=fovio_cmap)

    # Also register 4 colormaps (dark purple, light purple dark rose, yellow)
    list_names = ["Fovio_darkpurple", "Fovio_lightpurple", "Fovio_darkrose", "Fovio_yellow"]
    fovio_range = fovio_cmap([0, 0.25, 0.5, 1.0])
    # :3 because we keep only rgb, and remove alpha
    list_color_map = [CustomCmap(fovio_range[i][:3],
                                 fovio_range[i][:3],
                                 list_names[i]) for i in range(4)]

    for i in range(4):
        cmap = list_color_map[i]

        N = cmap.N
        cmap = cmap(np.arange(N))
        # Set alpha to go from 0.5 to 1
        cmap[:,-1] = np.linspace(0., 1., N)
        # Create new colormap with this alpha setting
        cmap = ListedColormap(cmap, name=list_names[i])
        # Register it
        plt.register_cmap(cmap=cmap)

    plt.rc("image", cmap="Fovio")
