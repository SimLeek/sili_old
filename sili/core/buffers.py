import struct

import kp
import numpy as np

from sili.core.serial import deserialize_buffer, serialize_buffer
from sili.core.devices.gpu import GPUManager

from typing import List


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


class GrayImageBuffer(object):
    def __init__(self, gpu: GPUManager, image):
        if isinstance(image, np.ndarray):
            # assume this is a numpy image from OpenCV:
            self.height = image.shape[0]
            self.width = image.shape[1]
            self.buffer = gpu.buffer(image)
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(image)}')

    def __setstate__(self, state):
        self.height, self.width = state[:2]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[3])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is picklable)
        return (self.height, self.width,
                serialize_buffer(self.buffer))

    @property
    def size(self):
        return self.height * self.width

    def set(self, image):
        if isinstance(image, np.ndarray):
            self.buffer.data()[...] = image.flatten()
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(image)}')

    def get(self):
        return self.buffer.data().reshape(self.height, self.width)


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
            self.height = array.shape[0]
            self.buffer = gpu.buffer(array)
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(array)}')

    def __setstate__(self, state):
        self.height = state[:2]
        # Restore the buffer from serialized data
        self.buffer = deserialize_buffer(state[2])

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is pickleable)
        return self.height, serialize_buffer(self.buffer)

    @property
    def size(self):
        return self.height

    def set(self, array):
        if isinstance(array, np.ndarray):
            self.buffer.data()[...] = array.flatten()
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(array)}')

    def get(self):
        return self.buffer.data().reshape([self.height])


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
        self.duplicates, = self.depth_in, self.depth_out = state[:3]
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


class MultiSquareMatrixBuffer(object):
    def __init__(self, gpu: GPUManager, matrices: List[np.ndarray]):
        self.num_matrices = len(matrices)
        self.matrix_sizes = [matrix.shape[0] for matrix in matrices]

        s = 0
        self.matrix_starts = [0]
        for m in self.matrix_sizes[:-1]:
            s += m * m
            self.matrix_starts.append(s)

        self.multi_matrix_buffer = gpu.buffer(
            np.concatenate(
                [m.flatten() for m in matrices]
            )
        )
        self.temp_buffer = gpu.buffer(
            np.concatenate(
                [np.eye(s) for s in self.matrix_sizes]
            )
        )

        self.i_pos_buffer = gpu.buffer(
            np.asarray([0], dtype=np.uint32).view(np.float32)
        )

        self.multi_mat_info_buffer = gpu.buffer(
            np.asarray([self.num_matrices] + list(zip(self.matrix_sizes, self.matrix_starts)), dtype=np.uint32).view(
                np.float32)
        )

    def to(self, t):
        if isinstance(t, GPUManager):
            if isinstance(self.multi_matrix_buffer, kp.Tensor):
                self.multi_matrix_buffer = self.multi_matrix_buffer.data()
            self.multi_matrix_buffer = t.buffer(self.multi_matrix_buffer)

            if isinstance(self.temp_buffer, kp.Tensor):
                self.temp_buffer = self.temp_buffer.data()
            self.temp_buffer = t.buffer(self.temp_buffer)

            if isinstance(self.multi_mat_info_buffer, kp.Tensor):
                self.multi_mat_info_buffer = self.multi_mat_info_buffer.data()
            self.multi_mat_info_buffer = t.buffer(self.multi_mat_info_buffer)

            if isinstance(self.i_pos_buffer, kp.Tensor):
                self.i_pos_buffer = self.i_pos_buffer.data()
            self.i_pos_buffer = t.buffer(self.i_pos_buffer)

        return self

    @property
    def size(self):
        return self.multi_matrix_buffer.size() * 2 + self.multi_mat_info_buffer.size()+1

    def set(self, matrices):
        if isinstance(matrices, list) and isinstance(matrices[0], np.ndarray):
            self.num_matrices = len(matrices)
            self.matrix_sizes = [matrix.shape[0] for matrix in matrices]

            s = 0
            self.matrix_starts = [0]
            for m in self.matrix_sizes[:-1]:
                s += m * m
                self.matrix_starts.append(s)

            self.multi_matrix_buffer.data()[...] = np.concatenate([m.flatten() for m in matrices])
            self.temp_buffer.data()[...] = np.concatenate([np.eye(s) for s in self.matrix_sizes])
            self.multi_mat_info_buffer.data()[...] = np.asarray(
                [self.num_matrices] + list(zip(self.matrix_sizes, self.matrix_starts)),
                dtype=np.uint32
            ).view(np.float32)
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(matrices)}')

    def get(self):
        mat_list = []
        flattened_array = self.multi_matrix_buffer.data()
        for si, st in zip(self.matrix_sizes, self.matrix_starts):
            mat_list.append(np.frombuffer(flattened_array[st:st + si * si]).reshape([si, si]))
        return mat_list

    def __setstate__(self, state):
        self.num_matrices, self.matrix_sizes, self.matrix_starts = state[:3]
        # Restore the buffer from serialized data
        self.multi_matrix_buffer = deserialize_buffer(state[3])

        self.temp_buffer = np.concatenate([np.eye(s) for s in self.matrix_sizes])

        self.multi_mat_info_buffer = np.asarray(
            [self.num_matrices] + list(zip(self.matrix_sizes, self.matrix_starts)),
            dtype=np.uint32
        ).view(np.float32)

        self.i_pos_buffer = np.asarray([0], dtype=np.uint32).view(np.float32)

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is picklable)
        return (
            self.num_matrices,
            self.matrix_sizes,
            self.matrix_starts,
            serialize_buffer(self.multi_matrix_buffer)
        )
