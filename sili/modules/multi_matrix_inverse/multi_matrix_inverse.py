import time

import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os
from sili.modules.base import Module

from sili.core.buffers import MultiSquareMatrixBuffer
from sili.core.devices.gpu import GPUManager, get_shader
from sili.core.util import find_good_dimension_sizes
file_path = os.path.dirname(os.path.abspath(__file__))

class MultiMatrixInverse(Module):
    def __init__(self,
                 gpu: GPUManager,
                 mat_buf: MultiSquareMatrixBuffer,
                 ):
        """
        Matrix inverse that works on multiple matrices at once.

        :param gpu: the GPUManager
        :param mat_buf: The multi matrix buffer
        """
        super().__init__(gpu)

        self.shad_norm = get_shader(file_path + os.sep + "normalize.comp")
        self.shad_gauss = get_shader(file_path + os.sep + "gauss_jordan.comp")
        self.shad_iter = get_shader(file_path + os.sep + "iter.comp")

        if not isinstance(mat_buf, MultiSquareMatrixBuffer):
            self.in_mat_buf = MultiSquareMatrixBuffer(self.gpu, mat_buf)
        else:
            self.in_mat_buf = mat_buf

        # PIPELINE OBJECTS:
        # FORWARD
        # these need to be accessible so that kompute can record input/output for pipelines
        self.forward_input_buffers = [self.in_mat_buf.multi_mat_info_buffer,
                                      self.in_mat_buf.temp_buffer,
                                      self.in_mat_buf.i_pos_buffer,
                                      self.in_mat_buf.multi_matrix_buffer]
        self.forward_output_buffers = [self.in_mat_buf.multi_matrix_buffer]

        algorithm_norm = self.gpu.manager.algorithm(
            self.forward_input_buffers,
            spirv=self.shad_norm,
            workgroup=[int(np.ceil(self.in_mat_buf.multi_matrix_buffer.size() / self.gpu.max_workgroup_invocations)), 0, 0],
            spec_consts=np.asarray([self.gpu.max_workgroup_invocations], dtype=np.uint32).view(
                np.float32
            ),
        )
        algorithm_gauss = self.gpu.manager.algorithm(
            self.forward_input_buffers,
            spirv=self.shad_gauss,
            workgroup=[int(np.ceil(self.in_mat_buf.multi_matrix_buffer.size() / self.gpu.max_workgroup_invocations)), 0, 0],
            spec_consts=np.asarray([self.gpu.max_workgroup_invocations], dtype=np.uint32).view(
                np.float32
            ),
        )
        algorithm_iter = self.gpu.manager.algorithm(
            [self.in_mat_buf.i_pos_buffer],
            spirv=self.shad_iter,
            workgroup=[1, 0, 0],
            spec_consts=np.asarray([1], dtype=np.uint32).view(np.float32),
        )

        self.forward_algorithms = [algorithm_norm, algorithm_gauss, algorithm_iter]

        self.has_forward = True
        self.has_backward = False
        self.has_optim = False

        self.basic_sequence = None

    def forward_ops(self):
        ops = []
        for i in range(max(self.in_mat_buf.matrix_sizes)):
            ops.extend([
                kp.OpAlgoDispatch(f) for f in self.forward_algorithms
            ])
        return ops

    def backward_ops(self):
        return []

    def optim_ops(self):
        return []

    def optim_buffs(self):
        return []

    def basic_forward(self, square_mat_buf:MultiSquareMatrixBuffer):
        # todo: make this a superclass method
        self.in_mat_buf.set(square_mat_buf.multi_matrix_buffer.data())
        if self.basic_sequence is None:
            self.basic_sequence = self.gpu.manager.sequence()
            self.basic_sequence.record(kp.OpTensorSyncDevice([*self.forward_input_buffers]))
            for f in self.forward_ops():
                self.basic_sequence.record(f)
            self.basic_sequence.record(kp.OpTensorSyncLocal([*self.forward_output_buffers]))
        self.t0 = time.time()
        self.basic_sequence.eval()
        self.t1 = time.time()
        return self.in_mat_buf

    def display_basic_forward_sequence(self, square_mat_buf:MultiSquareMatrixBuffer, display=None):
        if display is None:
            display = DirectDisplay()
        out_matrices = self.basic_forward(square_mat_buf)
        print(f"time:{self.t1-self.t0}, fps:{1./(self.t1-self.t0)}")
        for i, o in enumerate(out_matrices.get()):
            display.imshow(f'output {i}', o)
        while True:
            display.update()
            if display.window.is_closing:
                break



if __name__ == '__main__':

    gpu = GPUManager()
    im_pyr = pickle.load(f).to(gpu)

    pyr = PyrConv(gpu, im_pyr, lambda x: gaussian_center_surround(x, 3))
    pyr.display_basic_forward_sequence(im_pyr)

    #im_pyr = pyr.run_basic_forward_sequence(im)
    #with open("test_files/test_ai_pyr_pls_ignore.pyr", mode='wb') as f:
    #    pickle.dump(im_pyr, f)

