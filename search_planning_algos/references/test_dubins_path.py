import dubins
import matplotlib
import matplotlib.pyplot as plt
import numpy
import math

matplotlib.rcParams['figure.figsize'] = 12, 9

qs = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, numpy.pi / 4),
    (4.0, 4.0, numpy.pi / 4),
    (4.0, 0.0, 0.0),
    (-4.0, 0.0, 0.0),
    (4.0, 4.0, 0.0),
    (4.0, -4.0, 0.0),
    (-4.0, 4.0, 0.0),
    (-4.0, -4.0, 0.0),
    (4.0, 4.0, numpy.pi),
    (4.0, -4.0, numpy.pi),
    (0.5, 0.0, numpy.pi),
]

items = [
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 10),
    (0, 11),
    (1, 2),
    (2, 1)
]


def expand_axis(ax, scale, name):
    getter = getattr(ax, 'get_' + name)
    setter = getattr(ax, 'set_' + name)
    a, b = getter()
    mid = (a + b) / 2.0
    diff = (b - mid)
    setter(mid - scale * diff, mid + scale * diff)


def expand_plot(ax, scale=1.1):
    expand_axis(ax, scale, 'xlim')
    expand_axis(ax, scale, 'ylim')


"""['__new__', 'shortest_path', 'path', 'path_endpoint', 'path_length', 'segment_length', 'segment_length_normalized', 'path_type', 'sample', 'sample_many', 'extract_subpath', '__doc__', '__reduce__', '__setstate__', '__repr__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__init__', '__reduce_ex__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__']
"""


def plot_dubins_path(q0, q1, r=1.0, step_size=0.5):
    path = dubins.shortest_path(q0, q1, r)
    qs, ts = path.sample_many(step_size)
    qs = numpy.array(qs)
    xs = qs[:, 0]
    ys = qs[:, 1]
    us = xs + numpy.cos(qs[:, 2])
    vs = ys + numpy.sin(qs[:, 2])
    plt.plot(xs, ys, 'b-')
    plt.plot(xs, ys, 'r.')
    for i in range(qs.shape[0]):
        plt.plot([xs[i], us[i]], [ys[i], vs[i]], 'r-')
    ax = plt.gca()
    expand_plot(ax)
    ax.set_aspect('equal')


def plot_dubins_table(cols, rho=1.0):
    rows = ((len(items)) / cols) + 1
    for i, (a, b) in enumerate(items):
        plt.subplot(rows, cols, i + 1)
        plot_dubins_path(qs[a], qs[b], r=rho)
    plt.savefig('samples.png')
    plt.show()


# plot_dubins_table(3, 1.0)
start = [85, 40, -math.pi / 2]
goal = [85, 65, -math.pi / 2]
L = 1.0
max_steer = math.pi / 32
min_turn_rad = L / math.tan(max_steer)

plot_dubins_path(start, goal, r=min_turn_rad)
plt.show()
