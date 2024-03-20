import time

import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os

from sili.core.buffers import ImageBuffer, ImagePyramidBuffer, calculate_pyramid_levels, ConvDepthBuffer, \
    ConvDepthReductionBuffer
from sili.core.devices.gpu import GPUManager, get_shader
from sili.modules.base import Module

file_path = os.path.dirname(os.path.abspath(__file__))
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union


# note, forward shader order is:
# 	nnz_multi_reduction_1
#   nnz_multi_reduction_2
#   local_inclusive_scan
#   nonlocal_exclusive_scan
#   broadcast_array_add
#   local_bitonic_merge_sort


class SparseCSRBuffer(object):
    def __init__(self, gpu: GPUManager, max_nnz: int, dense_len: int):
        self.nnz: int = 0  # assuming uint32
        self.max_nnz: int = max_nnz
        self.dense_len: int = dense_len

        dtype = np.dtype([('pointer', np.uint32), ('index', np.uint32), ('value', np.float32)])
        sparse_vec = np.zeros([max_nnz], dtype=dtype)
        # off gpu or constants
        self.buffer = gpu.buffer(
            np.frombuffer(self.nnz.to_bytes(length=4, signed=False) + sparse_vec.tobytes(), dtype=np.float32))

    def __setstate__(self, state):
        self.max_nnz, self.dense_len = state[:3]
        self.nnz = int(np.frombuffer(self.buffer.data()[:1], dtype=np.uint32))  # assuming uint32
        # Restore the buffer from serialized data
        dtype = np.dtype([('pointer', np.uint32), ('index', np.uint32), ('value', np.float32)])
        self.buffer = deserialize_buffer(state[3][1:]).view(dtype)

    def __getstate__(self):
        # Return state to be pickled (excluding buffer, assuming buffer.data() is pickleable)
        return self.max_nnz, self.dense_len, serialize_buffer(self.buffer)

    @property
    def size(self):
        return self.max_nnz * 3 + 1

    def set(self, nnz: int, array: np.ndarray):
        if isinstance(array, np.ndarray):
            self.nnz = nnz
            (self.buffer.data()[:1]).frombytes(np.asarray([self.nnz], dtype=np.uint32).tobytes())
            self.buffer.data()[1:] = array.flatten()
        else:
            raise NotImplementedError(f'sorry, idk wtf this is: {type(array)}')

    def get(self):
        dtype = np.dtype([('pointer', np.uint32), ('index', np.uint32), ('value', np.float32)])
        return [
            int(np.frombuffer(self.buffer.data()[:1], dtype=np.uint32)),
            self.buffer.data()[1:].view(dtype),
        ]


class ToCSR(Module):
    def __init__(self,
                 gpu: GPUManager,
                 mat_io: kp.Tensor,
                 dense_size: (int,int),  # for buffers used for multiple things
                 csr_io=None,  # type: Union[SparseCSRBuffer, int]
                 intermediate_io: kp.Tensor = None,
                 forward=True,
                 backprop=False
                 ):
        """

        :param gpu: the GPUManager
        :param prediction_buffer: the kp.Tensor object
        :param forward: if true, actually calculate the forward error. (it's not necessary)
        :param backprop: if true, backprop error.
            Backpropagating error is the main reason to use this class, so default true.
        """
        self.gpu = gpu
        if not isinstance(mat_io, kp.Tensor):
            raise TypeError("vec_io Buffer must be a buffer attached to a device")
        else:
            self.mat_io = mat_io

        if not isinstance(intermediate_io, kp.Tensor):
            self.intermediate_io = None
        else:
            self.intermediate_io = intermediate_io

        self.dense_size = dense_size
        self.dense_len = dense_size[0]*dense_size[1]

        if isinstance(csr_io, int):
            # value -> index value
            self.csr_io = SparseCSRBuffer(self.gpu, csr_io, self.dense_len)
        elif isinstance(csr_io, SparseCSRBuffer):
            self.csr_io = csr_io
        else:
            raise TypeError("csr_io Buffer must be an int or SparseVectorBuffer. kp.buffer isn't set up yet.")

        self.has_forward = forward
        min_len = 1 + int(np.ceil(self.dense_len / self.gpu.max_workgroup_invocations))
        add = min_len
        while True:
            add = int(np.ceil((add - 1) / self.gpu.max_workgroup_invocations))
            if add == 1:
                break
            else:
                min_len += add
        if self.intermediate_io is None:
            self.intermediate_io = self.gpu.manager.tensor(np.zeros([min_len]))
        elif self.intermediate_io.size() < min_len:
            # note: you could use an intermediate vector... but there's really no point.
            #  It would just waste data since you could use it for the nnz buffer.
            raise TypeError(
                f"sparse vector isn't large enough to store intermediate sparsity calcualtion from full lendth vector. "
                f"{self.csr_io.buffer.size()}<{min_len}")

        if forward:
            self.forward_shader_1 = get_shader(file_path + os.sep + 'nnz_multi_reduction_1.comp')
            self.forward_shader_2 = get_shader(file_path + os.sep + 'nnz_multi_reduction_2.comp')
            self.forward_shader_3 = get_shader(file_path + os.sep + 'local_inclusive_scan.comp')
            self.forward_shader_4 = get_shader(file_path + os.sep + 'nonlocal_exclusive_scan.comp')
            self.forward_shader_5 = get_shader(file_path + os.sep + 'broadcast_array_add.comp')
            self.forward_shader_6 = get_shader(file_path + os.sep + 'local_bitonic_merge_sort.comp')

            # PIPELINE OBJECTS:
            # FORWARD
            # these need to be accessible so that kompute can record input/output for pipelines
            self.forward_input_buffers = [self.mat_io]
            self.forward_output_buffers = [self.intermediate_io, self.csr_io.buffer]

            # Note: I'm not going to sum the error into one number. It's kinda useful, but it costs a bit and you can do
            # that with a reduction op anywhere. Leaving it like this gives us options for nice error images, while
            # leaving options for making error graphs.
            # Note2: Currently, this reduces to num_workgroups in one step, which is a fair amount of reduction.
            # todo: Either make this return a full image, or do a full reduction to one number.
            self.forward_algorithms = []
            reduced_size = int(np.ceil(self.dense_len / min(self.dense_len, self.gpu.max_workgroup_invocations)))
            self.forward_algorithms.append(self.gpu.manager.algorithm(
                [self.mat_io, self.intermediate_io],
                spirv=self.forward_shader_1,
                workgroup=[reduced_size, 0, 0],
                spec_consts=np.asarray([min(self.dense_len, self.gpu.max_workgroup_invocations), self.dense_len],
                                       dtype=np.uint32).view(np.float32)
            ))

            initial_reduced_size = reduced_size
            initial_div = min(reduced_size, self.gpu.max_workgroup_invocations)

            div = initial_div
            reduced_size_2 = int(np.ceil(reduced_size / div))
            initial_reduced_size_2 = reduced_size_2
            out_loc = 1 + reduced_size
            in_loc = 1
            while True:
                if reduced_size_2 == 1:
                    out_loc = 0
                    self.forward_algorithms.append(self.gpu.manager.algorithm(
                        [self.intermediate_io],
                        spirv=self.forward_shader_2,
                        workgroup=[reduced_size_2, 0, 0],
                        spec_consts=np.asarray(
                            [div, in_loc, out_loc, reduced_size],
                            dtype=np.uint32).view(np.float32)
                    ))
                    break
                else:
                    # out_loc += reduced_size
                    self.forward_algorithms.append(self.gpu.manager.algorithm(
                        [self.intermediate_io],
                        spirv=self.forward_shader_2,
                        workgroup=[reduced_size_2, 0, 0],
                        spec_consts=np.asarray(
                            [div, in_loc, out_loc, reduced_size],
                            dtype=np.uint32).view(np.float32)
                    ))
                    in_loc = out_loc
                    reduced_size = reduced_size_2
                    div = min(reduced_size, self.gpu.max_workgroup_invocations)
                    reduced_size_2 = int(np.ceil(reduced_size / div))

            div = min(initial_reduced_size, self.gpu.max_workgroup_invocations)
            self.forward_algorithms.append(self.gpu.manager.algorithm(
                [self.intermediate_io],
                spirv=self.forward_shader_3,
                workgroup=[initial_reduced_size_2, 0, 0],
                spec_consts=np.asarray([div, initial_reduced_size, 1, 1],
                                       dtype=np.uint32).view(np.float32)
            ))

            # todo: loop this for orders of magnitude of max_workgroup_invocations
            #  this will likely require saving consecutive nnz reduction arrays and scans
            #  then broadcasting sums back to them level by level
            in_start_idx = 1 + (div - 1)
            out_start_idx = initial_reduced_size + 1
            div = min(initial_reduced_size_2, self.gpu.max_workgroup_invocations)
            reduce_size_3 = int(np.ceil(initial_reduced_size_2 / div))
            final_index = initial_reduced_size
            self.forward_algorithms.append(self.gpu.manager.algorithm(
                [self.intermediate_io],
                spirv=self.forward_shader_4,
                workgroup=[reduce_size_3, 0, 0],
                spec_consts=np.asarray(
                    [div, initial_reduced_size, initial_div, in_start_idx, out_start_idx, final_index],
                    dtype=np.uint32).view(np.float32)
            ))

            self.forward_algorithms.append(self.gpu.manager.algorithm(
                [self.intermediate_io],
                spirv=self.forward_shader_5,
                workgroup=[initial_reduced_size_2, 0, 0],
                spec_consts=np.asarray(
                    [in_start_idx, initial_reduced_size, in_start_idx, 1, out_start_idx],
                    dtype=np.uint32).view(np.float32)
            ))

            # note: intermediate_io and spvec_io should be seperate since intermediate_io has 1 index for every ~1000
            # indices in spvec_io, and would eventually overwrite what it's reading in that shader
            self.forward_algorithms.append(self.gpu.manager.algorithm(
                [self.mat_io, self.intermediate_io, self.csr_io.buffer],
                spirv=self.forward_shader_6,
                workgroup=[initial_reduced_size, 0, 0],
                spec_consts=np.asarray(
                    [in_start_idx, 1, *self.dense_size],
                    dtype=np.uint32).view(np.float32)
            ))

        self.has_backprop = backprop

        self.basic_sequence = None  # mostly for debugging

    def forward_ops(self):
        if self.has_forward:
            return [kp.OpAlgoDispatch(fa) for fa in self.forward_algorithms]
        else:
            return []

    def backward_ops(self):
        return []

    def optim_ops(self):
        return []

    def optim_buffs(self):
        # Doesn't make sense to optimize this. It isn't a neural net.
        return []

    def basic_forward(self):
        # todo: make this a superclass method
        if self.basic_sequence is None:
            self.basic_sequence = self.gpu.manager.sequence()
            self.basic_sequence.record(kp.OpTensorSyncDevice([*self.forward_input_buffers]))
            for f in self.forward_ops():
                self.basic_sequence.record(f)
            self.basic_sequence.record(kp.OpTensorSyncLocal([*self.forward_output_buffers]))
        self.basic_sequence.eval()
        return self.csr_io

    def display_basic_forward_sequence(self, shape=None, display=None):
        if display is None:
            display = DirectDisplay()
        err_np = self.basic_forward().data()
        if shape is not None:
            err_np = err_np.reshape(shape)
            min_err = np.min(err_np)
            max_err = np.max(err_np)
            display.imshow(f'output err', (err_np - min_err) / (max_err - min_err))
        else:
            raise ValueError("Cannot display a line that's likely larger than any screen.")
        while True:
            display.update()
            if display.window.is_closing:
                break


from sili.core.serial import deserialize_buffer, serialize_buffer

if __name__ == '__main__':
    in_array = np.zeros((2_000_000,))
    gpu = GPUManager()
    in_buf = gpu.buffer(in_array)
    to_spv = ToCSR(gpu, in_buf, 10000)

    indices = np.random.choice(2_000_000, 100)
    in_buf.data()[indices] = 1.0
    in_buf.data()[0] = 1.0
    t1 = time.time()
    to_spv.basic_forward()
    t2 = time.time()

    # todo: BUG. fix bug where the 0 indicies aren't in the right spot with ~9999 indices, or near full
    print(f"execution time = {t2 - t1}, or {1 / (t2 - t1)} fps")
    print(to_spv.csr_io.get()[1].view(np.uint32))

    # im_pyr = pyr.run_basic_forward_sequence(im)
    # with open("test_files/test_ai_pyr_pls_ignore.pyr", mode='wb') as f:
    #    pickle.dump(im_pyr, f)
