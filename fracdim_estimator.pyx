import math
from cpython cimport array
cimport cython

@cython.boundscheck(False)
def rotate(array.array points_x, array.array points_y, float angle):
    cdef float cos = math.cos(angle)
    cdef float sin = math.sin(angle)
    return (
        array.array('f',
            (x * cos + y * sin for x, y in zip(points_x, points_y))),
        array.array('f',
            (y * cos - x * sin for x, y in zip(points_x, points_y)))
    )

@cython.boundscheck(False)
def get_boxes_filled(array.array points_x, array.array points_y, float anchor_x,
    float anchor_y, float box_size):
    return set(
        (int(round((x - anchor_x) / box_size)),
        int(round((y - anchor_y) / box_size)))
            for x, y in zip(points_x, points_y))
