import contextlib
import math
import random
import time
from collections import defaultdict
from multiprocessing import Pool

import cv2
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

CODES = [Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY
]

MAX_ANGLE = math.pi / 2
ROTATABLE_FACTOR = math.sqrt(2)
STARTING_DIM = 8
ANGLE_TRIALS = 2
PLACEMENT_TRIALS = 10
NUM_PROCESSES = 4


@contextlib.contextmanager
def process_pool(num_processes):
    '''
    Context manager to ensure processing pools are closed when they go out
    of scope.

    :param num_processes: number of processes in the pool

    Usage:

    with process_pool(4) as pool:
        [do stuff with pool]
    '''
    pool = Pool(processes=num_processes)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()


def make_rotation_matrix(angle):
    return np.array([[math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]])


def rotate(point_set, angle=None):
    if angle is None:
        angle = random.random() * MAX_ANGLE
    rotation_matrix = make_rotation_matrix(angle)
    return angle, point_set.dot(rotation_matrix)


def get_boxes_filled(point_set, anchor_x, anchor_y, box_size):
    box_points = (point_set - (anchor_x, anchor_y)) / box_size
    return set((int(x + 0.5), int(y + 0.5)) for x, y in box_points[:, 0])


def estimate_min_boxes_filled(point_set, initial_box_size, box_dim, box_size):
    metrics = defaultdict(float)
    min_num_boxes_filled = len(point_set) + 1
    min_boxes_filled = None
    theta_min, anchor_min = 0, (0.0, 0.0)
    start_rotate_secs = time.time()
    for i in xrange(ANGLE_TRIALS):
        theta, rotated = rotate(point_set, angle=0.2)
        start_placement_secs = time.time()
        metrics['rotate_secs'] += start_placement_secs - start_rotate_secs
        for j in xrange(PLACEMENT_TRIALS):
            anchor = [(0.5 - random.random()) * initial_box_size,
                (0.5 - random.random()) * initial_box_size]
            anchor = [0.0, 0.0]
            start_count_secs = time.time()
            metrics['placement_secs'] += start_count_secs - start_placement_secs
            boxes_filled = get_boxes_filled(rotated, anchor[0], anchor[1],
                box_size)
            start_placement_secs = time.time()
            metrics['count_secs'] += start_placement_secs - start_count_secs
            num_boxes_filled = len(boxes_filled)
            if num_boxes_filled < min_num_boxes_filled:
                min_num_boxes_filled = num_boxes_filled
                min_boxes_filled = boxes_filled
                theta_min = theta
                anchor_min = anchor
        start_rotate_secs = time.time()
        metrics['placement_secs'] += start_rotate_secs - start_placement_secs
    return theta_min, anchor_min, box_dim, box_size, min_num_boxes_filled, min_boxes_filled, metrics


class FractalDimensionEstimator(object):
    def __init__(self, image):
        self.image = image
        self.point_set = cv2.findNonZero(self.image)
        offsets = (self.point_set.max(0) - self.point_set.min(0)) / 2
        self.point_set -= offsets
        self.initial_box_size = int(math.ceil(ROTATABLE_FACTOR * (
            self.point_set.max(0) - self.point_set.min(0)).max()))
        s = (self.point_set.max(0) - self.point_set.min(0)).max() / 2
        mpl.rcParams['toolbar'] = 'None'
        self.figure, self.ax = plt.subplots()
        self.ax.plot([-s, -s, s, s, -s], [-s, s, s, -s, -s])
        self.lines_pts, = self.ax.plot([], [], '.', color='gray', alpha=0.5)
        self.lines_pts2, = self.ax.plot([], [], '.', color='red')
        plt.axis('equal')
        self.patches = []

    def show_result(self, point_set, anchor, boxes_filled, box_size, theta):
        self.lines_pts.set_xdata(point_set[:, 0, 0])
        self.lines_pts.set_ydata(point_set[:, 0, 1])
        if anchor:
            for patch in self.patches:
                patch.remove()
            self.patches = []
            t = Affine2D().scale(box_size).translate(anchor[0] - box_size,
                anchor[1] - box_size).rotate_deg(-180.0 / math.pi * theta)
            xx = []
            yy = []
            for x, y in boxes_filled:
                pts = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y)]
                verts = t.transform(pts)
                xx.append(x * box_size)
                yy.append(y * box_size)
                path = Path(verts, CODES)
                patch = patches.PathPatch(path, facecolor='yellow', lw=1,
                    edgecolor='black', alpha=0.5)
                self.patches.append(self.ax.add_patch(patch))

            self.lines_pts2.set_xdata(xx)
            self.lines_pts2.set_ydata(yy)
            plt.axis('equal')
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            return

    def est_frac_dim(self):
        box_dim = STARTING_DIM
        box_size = float(self.initial_box_size / box_dim)
        max_dim = int(math.ceil(STARTING_DIM * math.pow(2.0,
            math.log10(len(self.point_set))) / 2))
        print('Image size={}'.format(self.image.shape))
        print('Number of points={:,}'.format(len(self.point_set)))
        print('Boxes\tBox Dim\t\t(X0, Y0)\t\tBox Size\t\tTheta')
        box_dims = []
        box_fills = []
        self.show_result(self.point_set, None, None, None, None)
        total_metrics = defaultdict(float)
        with process_pool(NUM_PROCESSES) as pool:
            results = []
            while box_dim <= max_dim + 0.0001:
                results.append(pool.apply_async(estimate_min_boxes_filled,
                    (self.point_set, self.initial_box_size, box_dim, box_size)))
                box_dim = box_dim * 2.0
                box_size = box_size / 2.0

            for result in results:
                theta, anchor, box_dim, box_size, num_boxes_filled, boxes_filled, metrics = result.get()
                print(
                    '{}\t\t1/{}\t\t({:6.1f}, {:6.1f})\t{:9.1f}\t{:9.1f}'.format(
                        num_boxes_filled, box_dim, anchor[0], anchor[1],
                        box_size,
                        180.0 / math.pi * theta))
                box_dims.append(math.log(box_dim))
                box_fills.append(math.log(num_boxes_filled))
                _, rotated = rotate(self.point_set, theta)
                self.show_result(rotated, anchor, boxes_filled, box_size, theta)
                for k in metrics:
                    total_metrics[k] += metrics[k]

        frac_dim, _, _, _, _ = stats.linregress(box_dims, box_fills)
        print('\n'.join(
            '{}: {}'.format(k, v) for k, v in total_metrics.iteritems()))
        return frac_dim


if __name__ == '__main__':
    plt.ion()
    start_secs = time.time()
    img = cv2.imread('data/leaf.jpg', cv2.IMREAD_GRAYSCALE)
    _, b_img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    estimator = FractalDimensionEstimator(b_img)
    print('Estimated fractal dimension={:.2f} in {} secs'.format(
        estimator.est_frac_dim(),
        time.time() - start_secs))
