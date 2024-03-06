import struct

import kp
import numpy as np

from sili.core.serial import deserialize_buffer, serialize_buffer
from sili.core.devices.gpu import GPUManager


class ImageBuffer(object):
    def __init__(self, gpu: GPUManager, image):
        if isinstance(image, np.ndarray):
            # assume this is a numpy image from OpenCV:
            self.height = image.shape[0]
            self.width = image.shape[1]
            self.colors = image.shape[2]
            self.buffer = gpu.buffer(image)
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(image)}')

    def __setstate__(self, state):
        self.height, self.width, self.colors = state[:3]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[3])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is picklable)
        return (self.height, self.width, self.colors,
                serialize_buffer(self.buffer))

    @property
    def size(self):
        return self.height * self.width * self.colors

    def set(self, image):
        if isinstance(image, np.ndarray):
            self.buffer.data()[...] = image.flatten()
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(image)}')

    def get(self):
        return self.buffer.data().reshape(self.height, self.width, self.colors)


class ConvDepthBuffer(object):
    def __init__(self, gpu: GPUManager, array):
        if isinstance(array, np.ndarray):
            # assume this is a numpy image from OpenCV:
            self.depth_in = array.shape[0]
            self.depth_out = array.shape[1]
            self.buffer = gpu.buffer(array)
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(array)}')

    def __setstate__(self, state):
        self.depth_in, self.depth_out = state[:2]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[2])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is pickleable)
        return self.depth_in, self.depth_out, serialize_buffer(self.buffer)

    @property
    def size(self):
        return self.depth_in * self.depth_out

    def set(self, array):
        if isinstance(array, np.ndarray):
            self.buffer.data()[...] = array.flatten()
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(array)}')

    def get(self):
        return self.buffer.data().reshape(self.depth_in, self.depth_out)


class ConvVertPyrBuffer(object):
    def __init__(self, gpu: GPUManager, array):
        if isinstance(array, np.ndarray):
            # assume this is a numpy image from OpenCV:
            self.levels = array.shape[0]
            self.height = array.shape[1]
            self.buffer = gpu.buffer(array)
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(array)}')

    def __setstate__(self, state):
        self.levels, self.height = state[:2]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[2])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is pickleable)
        return self.levels, self.height, serialize_buffer(self.buffer)

    @property
    def size(self):
        return self.levels * self.height

    def set(self, array):
        if isinstance(array, np.ndarray):
            self.buffer.data()[...] = array.flatten()
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(array)}')

    def get(self):
        return self.buffer.data().reshape(self.levels, self.height)


class ConvDepthReductionBuffer(object):
    """Useful for running compute reductions for determining updates"""
    def __init__(self, gpu: GPUManager, array):
        if isinstance(array, np.ndarray):
            # assume this is a numpy image from OpenCV:
            self.duplicates = array.shape[0]
            self.depth_in = array.shape[1]
            self.depth_out = array.shape[2]
            self.buffer = gpu.buffer(array)
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(array)}')

    def __setstate__(self, state):
        self.duplicates, =self.depth_in, self.depth_out = state[:3]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[3])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is pickleable)
        return self.duplicates, self.depth_in, self.depth_out, serialize_buffer(self.buffer)

    @property
    def size(self):
        return self.duplicates * self.depth_in * self.depth_out

    def set(self, array):
        if isinstance(array, np.ndarray):
            self.buffer.data()[...] = array.flatten()
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(array)}')

    def get(self):
        return self.buffer.data().reshape(self.duplicates, self.depth_in, self.depth_out)

class ImagePyramidBuffer(object):
    def __init__(self, gpu: GPUManager, levels, channels=3, use_lvl_buf=True):
        self.levels = levels
        self.channels = channels
        size = channels * (self.levels[-3] + 1)  # start, w, h sequence. last w,h is 1x1, so last start is size-1.
        self.image_buffer = gpu.buffer(np.zeros((size,)))

        levels_str = struct.pack(f'={len(levels)}i', *levels)
        levels_glsl = np.frombuffer(levels_str, dtype=np.float32)
        self.use_lvl_buf = use_lvl_buf
        if self.use_lvl_buf:
            self.pyr_lvl_buffer = gpu.buffer(levels_glsl)
        else:
            self.pyr_lvl_buffer = None

    def to(self, t):
        if isinstance(t, GPUManager):
            if isinstance(self.image_buffer, kp.Tensor):
                self.image_buffer = self.image_buffer.data()
            self.image_buffer = t.buffer(self.image_buffer)
            if self.use_lvl_buf:
                if isinstance(self.pyr_lvl_buffer, kp.Tensor):
                    self.pyr_lvl_buffer = self.pyr_lvl_buffer.data()
                self.pyr_lvl_buffer = t.buffer(self.pyr_lvl_buffer)
        return self

    @property
    def size(self):
        return self.image_buffer.size()

    def set(self, image):
        if isinstance(image, np.ndarray):
            self.image_buffer.data()[...] = image.flatten()
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(image)}')

    def get(self):
        im_list = []
        flattened_array = self.image_buffer.data()
        for l in range(2, len(self.levels), 3):
            start, width, height = int(self.levels[l]), int(self.levels[l + 1]), int(self.levels[l + 2]),
            end_idx = int(start * self.channels + (width * height * self.channels))

            image = flattened_array[int(start * self.channels):end_idx].reshape(width, height, self.channels)
            im_list.append(image)
        return im_list

    def __setstate__(self, state):
        self.levels, self.channels, self.use_lvl_buf = state[:3]
        # Restore the buffer from serialized data
        self.image_buffer = deserialize_buffer(state[3])
        if self.use_lvl_buf:
            self.pyr_lvl_buffer = deserialize_buffer(state[4])
        else:
            self.pyr_lvl_buffer = None

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is picklable)
        return (self.levels, self.channels, self.use_lvl_buf,
                serialize_buffer(self.image_buffer),
                serialize_buffer(self.pyr_lvl_buffer) if self.use_lvl_buf else None)


def calculate_pyramid_levels(h, w, s, c=3):
    levels = []
    start_idx = 0

    while h > 1 or w > 1:
        levels.extend([
            start_idx,
            h,
            w
        ])

        start_idx += h * w
        h = int(max(1, h / s))
        w = int(max(1, w / s))

    levels.extend([
        start_idx,
        1,
        1
    ])
    levels = [c, int(len(levels) // 3)] + levels

    return levels
