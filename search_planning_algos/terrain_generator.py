import numpy as np
import scipy.io
import yaml
import matplotlib.pyplot as plt
import os


DEFAULT_ENV_CONFIG = "/Users/Alvin/Documents/Code/safety_reachability_AV_research/assured_autonomy_car/search_planning_algos/terrain_params.yaml"


def save_to_matfile(arr_dict, filename):
    """[summary]

    Args:
        arr_dict (dict): dict mapping array names to arrays
        filename ([type]): [description]
    """
    if not os._exists(filename):
        open(filename, 'a').close()
    scipy.io.savemat(filename, arr_dict)


def test1_terrain(env_config_file):
    """Generates terrain for first test case:
    Robot initially thinks entire terrain is flat with z=0. However,
    there is a square object sitting in the middle. Robot should initially
    plan a straight path from any start to end across the terrain, but
    will have to move around the obstacle as it observes obstacle.
    """
    # load in params matching what MATLAB sim expects
    with open(env_config_file, 'r') as f:
        params = yaml.load(f)
    cellsize = params["cellsize"]
    cells_per_m = int(1.0 / cellsize)
    map_width_cont = params["map_width_cont"]
    map_length_cont = params["map_length_cont"]

    # generate empty map
    disc_width = int(map_width_cont / cellsize)
    disc_length = int(map_length_cont / cellsize)
    new_map = np.zeros(shape=(disc_length, disc_width), dtype=float)

    # populate obstacles and features
    # obstacle making up interval of 30% to 70% of width and length
    obs_height = 5
    obst_x_bounds = np.array([0.35, 0.65]) * map_width_cont / cellsize
    obst_y_bounds = np.array([0.4, 0.5]) * map_length_cont / cellsize
    obst_x_bounds = obst_x_bounds.astype(int)
    obst_y_bounds = obst_y_bounds.astype(int)
    new_map[obst_y_bounds[0]:obst_y_bounds[1],
            obst_x_bounds[0]: obst_x_bounds[1]] = obs_height

    # MUST transopse map because MATLAB takes in col-major
    new_map = new_map.T

    # MATLAB calls GridSurf(xmin,xmax,nx,ymin,ymax,ny,Z)
    xbounds = [0, map_width_cont]
    ybounds = [0, map_length_cont]
    nx, ny = disc_width, disc_length

    # save into file
    filename = "terrains/test1_terrain.mat"
    data = dict(map=new_map, xbounds=xbounds, ybounds=ybounds, nx=nx, ny=ny)
    save_to_matfile(data, filename)

    print("%s saved!" % filename)


def mousePressed(event, data, ax):
    cx, cy = int(event.xdata), int(event.ydata)
    try:
        data.map[cy - data.radius: cy + data.radius,
                 cx - data.radius: cx + data.radius] += data.update
    except:
        for dy in range(-data.radius, data.radius):
            for dx in range(-data.radius, data.radius):
                if (0 <= cx + dx < data.width) and (0 <= cy + dy < data.height):
                    data.map[cy + dy, cx + dx] += data.update[
                        dy + data.radius, dx + data.radius]

    print(data.map[cy - data.radius: cy + data.radius,
                   cx - data.radius: cx + data.radius])
    plt.clf()
    plt.imshow(data.map)
    plt.draw()


def keyPressed(event, data):
    if event.key == " ":
        data.update *= -1
    if event.key == "up":
        data.radius += 1
        new_update(data)
    elif event.key == "down":
        if data.radius > 1:
            data.radius -= 1
            new_update(data)


def new_update(data):
    try:
        is_neg = (np.sum(data.update) < 0)
    except AttributeError:
        # if update window doesn't exist yet
        is_neg = False
    data.update = np.zeros((2 * data.radius, 2 * data.radius))
    for dy in range(-data.radius, data.radius):
        for dx in range(-data.radius, data.radius):
            if np.linalg.norm([dy, dx]) <= data.radius:
                data.update[dy + data.radius, dx +
                            data.radius] = data.update_incr

    if is_neg:
        data.update *= -1


def draw_custom(env_config_file):
    class Data():
        pass

    # load in params matching what MATLAB sim expects
    with open(env_config_file, 'r') as f:
        params = yaml.load(f)
    cellsize = params["cellsize"]
    cells_per_m = int(1.0 / cellsize)
    map_width_cont = params["map_width_cont"]
    map_length_cont = params["map_length_cont"]
    update_incr = params["update_height_increment"]
    xbounds = [0, map_width_cont]
    ybounds = [0, map_length_cont]

    # generate empty map
    disc_width = int(map_width_cont / cellsize)
    disc_length = int(map_length_cont / cellsize)
    data = Data()
    data.width = disc_width
    data.height = disc_length
    data.map = np.ones(shape=(disc_length, disc_width), dtype=float)
    data.radius = 20
    data.update_incr = update_incr
    new_update(data)

    # plt.ion()
    fig, ax = plt.subplots()
    data_plot = ax.imshow(data.map)

    def mousePressedCallback(event):
        mousePressed(event, data, ax)

    def keyPressedCallback(event):
        keyPressed(event, data)

    fig.canvas.mpl_connect('button_press_event', mousePressedCallback)
    fig.canvas.mpl_connect('key_press_event', keyPressedCallback)

    plt.show()
    plt.draw()

    # save into file
    filename = "search_planning_algos/terrains/terrain1.mat"
    results = dict(map=data.map, xbounds=xbounds,
                   ybounds=ybounds, nx=disc_width, ny=disc_length)
    save_to_matfile(results, filename)


if __name__ == "__main__":
    # test1_terrain(DEFAULT_ENV_CONFIG)
    draw_custom(DEFAULT_ENV_CONFIG)
