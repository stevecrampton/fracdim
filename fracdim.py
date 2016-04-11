import array
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
import pyximport
import scipy.stats as stats
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

pyximport.install(pyimport=True)

from fracdim_estimator import get_boxes_filled, rotate

CODES = [Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY
]

ROTATABLE_FACTOR = math.sqrt(2)
MAX_ANGLE = math.pi / 2
STARTING_DIM = 8
ANGLE_TRIALS = 10
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


def rotate_old(point_set, angle=None):
    if angle is None:
        angle = random.random() * MAX_ANGLE
    rotation_matrix = make_rotation_matrix(angle)
    return angle, point_set.dot(rotation_matrix)


def estimate_min_boxes_filled(xx, yy, initial_box_size, box_dim, box_size):
    metrics = defaultdict(float)
    min_num_boxes_filled = len(xx) + 1
    min_boxes_filled = None
    theta_min, anchor_min = 0, (0.0, 0.0)
    start_rotate_secs = time.time()
    for i in xrange(ANGLE_TRIALS):
        theta = random.random() * MAX_ANGLE
        theta = 0.0
        xx_r, yy_r = rotate(xx, yy, angle=0)
        start_placement_secs = time.time()
        metrics['rotate_secs'] += start_placement_secs - start_rotate_secs
        for j in xrange(PLACEMENT_TRIALS):
            anchor = [(0.5 - random.random()) * initial_box_size,
                (0.5 - random.random()) * initial_box_size]
            anchor = [0.0, 0.0]
            start_count_secs = time.time()
            metrics['placement_secs'] += start_count_secs - start_placement_secs
            boxes_filled = get_boxes_filled(xx_r, yy_r, anchor[0], anchor[1],
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
    def __init__(self, xx, yy):
        '''
        :param xx: x coordinates of point set
        :param yy: y coordinates of point set
        '''
        # center the point set on the origin
        x_range = max(xx) - min(xx)
        y_range = max(yy) - min(yy)
        self.xx = array.array('f', (x - x_range / 2.0 for x in xx))
        self.yy = array.array('f', (y - y_range / 2.0 for y in yy))
        # the initial box size is based on the smallest box that would contain
        # all rotations of the point set
        max_range = max(x_range, y_range)
        self.initial_box_size = math.ceil(max(ROTATABLE_FACTOR * max_range, 1.0))
        # set up the visualization window
        mpl.rcParams['toolbar'] = 'None'
        self.figure, self.ax = plt.subplots()
        s = int(math.ceil(self.initial_box_size / 2.0))
        self.ax.plot([-s, -s, s, s, -s], [-s, s, s, -s, -s])
        self.lines_pts, = self.ax.plot([], [], '.', color='gray', alpha=0.5)
        self.lines_pts2, = self.ax.plot([], [], '.', color='red')
        plt.axis('equal')
        # pyplot patches will hold visual representations of the boxes
        self.patches = []
        # draw the point set
        self.show_result(self.xx, self.yy, None, None, None, None)

    def show_result(self, boxes_x, boxes_y, anchor=None, boxes_filled=None,
        box_size=None, theta=None):
        '''
        Update the display window with the latest iteration
        :param boxes_x: x coordinates of the point set
        :param boxes_y: y coordinates of the point set
        :param anchor: the anchor point used for the boxes
        :param boxes_filled: the set of boxes filled by at least one point
        :param box_size: the box size in pixels
        :param theta: the angle of the boxes
        '''
        # redraw the point set (so it can be displayed in rotated form)
        self.lines_pts.set_xdata(boxes_x)
        self.lines_pts.set_ydata(boxes_y)

        if anchor:
            # if an anchor was provided, show the boxes
            # remove the patches from the previous display (if any)
            for patch in self.patches:
                patch.remove()
            self.patches = []
            # build an affine transform to transform the boxes, based on the
            # anchor point, box angle, and box size
            transformer = Affine2D() \
                .scale(box_size) \
                .translate(anchor[0] - box_size, anchor[1] - box_size) \
                .rotate_deg(-180.0 / math.pi * theta)
            # the coordinates of one corner of the boxes
            boxes_x = []
            boxes_y = []
            for x, y in boxes_filled:
                # create a patch for each box
                pts = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y)]
                verts = transformer.transform(pts)
                boxes_x.append(x * box_size)
                boxes_y.append(y * box_size)
                path = Path(verts, CODES)
                patch = patches.PathPatch(path, facecolor='yellow', lw=1,
                    edgecolor='black', alpha=0.5)
                self.patches.append(self.ax.add_patch(patch))
            # display the boxes
            self.lines_pts2.set_xdata(boxes_x)
            self.lines_pts2.set_ydata(boxes_y)
            plt.axis('equal')
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

    def estimate(self):
        '''
        Estimate the fractal dimension of the point set
        :return: fractal dimension (between 0.0 and 2.0)
        '''
        num_points = len(self.xx)
        print('Number of points={:,}'.format(num_points))
        if num_points == 1:
            return 0.0

        box_dim = STARTING_DIM
        box_size = self.initial_box_size / box_dim
        max_dim = int(math.ceil(STARTING_DIM *
                                math.pow(2.0, math.log10(num_points)) / 2))
        print('Boxes\tBox Dim\t\t(X0, Y0)\tBox Size\tTheta')
        box_dims = []
        box_fills = []
        total_metrics = defaultdict(float)
        with process_pool(NUM_PROCESSES) as pool:
            results = []
            while box_dim <= max_dim:
                results.append(pool.apply_async(estimate_min_boxes_filled,
                    (self.xx, self.yy, self.initial_box_size, box_dim,
                    box_size)))
                box_dim = box_dim * 2
                box_size = box_size / 2.0

            for result in results:
                theta, anchor, box_dim, box_size, num_boxes_filled, boxes_filled, metrics = result.get()
                print(
                    '{}\t1/{}\t\t({:5.1f}, {:5.1f})\t{:9.1f}\t{:9.1f}'.format(
                        num_boxes_filled, box_dim, anchor[0], anchor[1],
                        box_size,
                        180.0 / math.pi * theta))
                box_dims.append(math.log(box_dim))
                box_fills.append(math.log(num_boxes_filled))
                xx_r, yy_r = rotate(self.xx, self.yy, theta)
                self.show_result(xx_r, yy_r, anchor, boxes_filled, box_size,
                    theta)
                for k in metrics:
                    total_metrics[k] += metrics[k]

        frac_dim, _, _, _, _ = stats.linregress(box_dims, box_fills)
        print('Metrics:')
        print('\n'.join(
            '{:15s} {:8.2f}'.format(k, v) for k, v in total_metrics.iteritems()))
        return frac_dim


if __name__ == '__main__':
    plt.ion()
    start_secs = time.time()
    img = cv2.imread('data/leaf.jpg', cv2.IMREAD_GRAYSCALE)
    _, b_img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    point_set = cv2.findNonZero(b_img)
    xx = array.array('f', point_set[:, 0, 0])
    yy = array.array('f', point_set[:, 0, 1])
    estimator = FractalDimensionEstimator(xx, yy)
    print('Image size={}'.format(img.shape))
    print('Estimated fractal dimension of {:.2f} in {:.2f} secs'.format(
        estimator.estimate(),
        time.time() - start_secs))
