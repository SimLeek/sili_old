import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os

from sili.core.buffers import ImageBuffer, ImagePyramidBuffer, calculate_pyramid_levels, ConvDepthBuffer, ConvDepthReductionBuffer
from sili.core.devices.gpu import GPUManager, get_shader

file_path = os.path.dirname(os.path.abspath(__file__))



def depth_edge_matrix(n):
    # Create the initial matrix A
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = abs(i - j)

    # Normalize each column of A by its sum to create matrix B
    column_sums = A.sum(axis=0)
    B = A / column_sums

    # Create the identity matrix of size n
    identity = np.eye(n)

    # Return the result: identity matrix minus B
    return (identity - B)*2

def one_hot_mat(n, x, y):
    A = np.zeros((n, n))
    A[x,y] = 1
    return A

class DepthPyrConv(object):
    def __init__(self,
                 gpu: GPUManager,
                 image_pyr: ImagePyramidBuffer,
                 init_conv = None,
                 backprop_input_buf = None,
                 backprop_conv = False):
        """

        :param gpu: the GPUManager
        :param image_pyr: Either an ImagePyramidBuffer or a list containing the widths and heights of all images in pyr
        """
        self.reverse = False  # set this to true for pipeline creators to only use it for backwards
        # self.symmetric = True  # conv and conv transpose is the same since the transpose is learned

        self.gpu = gpu
        self.forward_shader = get_shader(file_path + os.sep + 'depth_pyr_forward.comp')
        if not isinstance(image_pyr, ImagePyramidBuffer):
            self.in_img_pyr = ImagePyramidBuffer(self.gpu, image_pyr)
        else:
            self.in_img_pyr = image_pyr

        self.out_pyr = ImagePyramidBuffer(gpu, self.in_img_pyr.levels, use_lvl_buf=False)
        if callable(init_conv):
            self.depth_conv = ConvDepthBuffer(gpu, init_conv(self.in_img_pyr.levels[1]))
        else:
            self.depth_conv = ConvDepthBuffer(gpu, np.zeros((self.in_img_pyr.levels[1], self.in_img_pyr.levels[1])))
        self.depth_conv_str = ConvDepthBuffer(gpu, np.zeros((self.in_img_pyr.levels[1], self.in_img_pyr.levels[1])))  # weight strengths, for adagrad

        # PIPELINE OBJECTS:
        # FORWARD
        # these need to be accessible so that kompute can record input/output for pipelines
        self.forward_input_buffers = [self.in_img_pyr.image_buffer, self.in_img_pyr.pyr_lvl_buffer, self.depth_conv.buffer]
        self.forward_output_buffers = [self.out_pyr.image_buffer]

        # todo: this may be a more general superclass method
        # this is the part that gets shoved into popelines
        self.forward_algorithm = self.gpu.manager.algorithm(
            [self.in_img_pyr.image_buffer, self.in_img_pyr.pyr_lvl_buffer, self.out_pyr.image_buffer, self.depth_conv.buffer],
            spirv=self.forward_shader,
            workgroup=[int(np.ceil(self.in_img_pyr.size / self.gpu.max_workgroup_invocations)), 0, 0],
            spec_consts=np.asarray([self.gpu.max_workgroup_invocations], dtype=np.uint32).view(np.float32)
        )

        self.has_forward = True
        self.has_backward = False
        self.has_optim = False

        # BACKWARD:
        self.out_pyr_err = ImagePyramidBuffer(gpu, self.in_img_pyr.levels, use_lvl_buf=False)


        # this is input to the backwards op, so input should come from later parts of the pipeline
        self.backward_input_buffers = []
        self.backward_output_buffers = []
        self.backprop_input = False
        self.backprop_conv = False
        if backprop_input_buf is not None:
            self.has_backward = True
            self.backprop_input = True
            shad_input_back = get_shader(file_path + os.sep + 'depth_pyr_backward_input.comp')

            self.backward_input_buffers = [self.out_pyr.image_buffer, self.in_img_pyr.pyr_lvl_buffer, self.depth_conv.buffer, self.out_pyr_err]
            self.backward_output_buffers = [backprop_input_buf]

            self.algorithm_input_back = self.gpu.manager.algorithm(
                [*self.backward_input_buffers, *self.backward_output_buffers],
                spirv=shad_input_back,
                workgroup=[int(np.ceil(self.out_pyr.image_buffer.size() / self.gpu.max_workgroup_invocations)), 0, 0],
                spec_consts=np.asarray([self.gpu.max_workgroup_invocations], dtype=np.uint32).view(np.float32)
            )

        if backprop_conv:
            self.has_backward = True
            self.backprop_conv = True
            shad_conv_back_prepool_1 = get_shader(file_path + os.sep + 'depth_pyr_backward_conv_prepool.comp')
            shad_conv_back_prepool_2 = get_shader(file_path + os.sep + 'index2_value2_max_reduction.comp')
            shad_conv_back = get_shader(file_path + os.sep + 'depth_pyr_backward_conv.comp')
            shad_conv_reduce = get_shader(file_path + os.sep + 'conv_err_sum_reduce.comp')

            # private buffers
            buf_conv_prepool = self.gpu.manager.tensor(np.zeros([int(np.ceil(self.in_img_pyr.size * 4 / self.gpu.max_workgroup_invocations))]))


            # no special backwards input/output buffers, because the backprop ends in the conv buffer inside this object
            self.algorithm_conv_prepool_1 = self.gpu.manager.algorithm(
                [self.in_img_pyr.image_buffer, self.out_pyr_err.image_buffer, buf_conv_prepool, self.in_img_pyr.pyr_lvl_buffer, self.depth_conv.buffer,
                 self.depth_conv_str.buffer],
                spirv=shad_conv_back_prepool_1,
                workgroup=[int(np.ceil(self.in_img_pyr.size * self.in_img_pyr.levels[1] / self.gpu.max_workgroup_invocations)), 0, 0],
                # workgroup=[1, 0, 0],
                spec_consts=np.asarray([self.gpu.max_workgroup_invocations], dtype=np.uint32).view(np.float32)
            )

            self.conv_prepool_reductions = []
            last_s0 = prev_s0 = int(buf_conv_prepool.size() / 4)
            s0 = int(np.ceil(buf_conv_prepool.size() / 4 / self.gpu.max_workgroup_invocations))
            while True:
                if s0 <= 1:  # do while loop to ensure we include the final reduce
                    break
                self.conv_prepool_reductions.append(
                    self.gpu.manager.algorithm(
                        [buf_conv_prepool, self.depth_conv_str.buffer],
                        spirv=shad_conv_back_prepool_2,
                        workgroup=[last_s0, 0, 0],
                        spec_consts=np.asarray([s0, prev_s0, self.in_img_pyr.levels[1]], dtype=np.uint32).view(np.float32)
                    )
                )
                last_s0 = int(np.ceil(prev_s0 / s0))
                prev_s0 = s0
                s0 = int(np.ceil(s0 / 2))

            memory_sensitive_divisor = int(
                2 ** int(np.log2(self.gpu.maxComputeSharedMemorySize / (
                            self.in_img_pyr.levels[1] * self.in_img_pyr.levels[1] * 4 * 2))))

            self.depth_conv_err = ConvDepthReductionBuffer(gpu, np.zeros(
                (int(np.ceil(last_s0/memory_sensitive_divisor)), self.in_img_pyr.levels[1], self.in_img_pyr.levels[1])))
            self.depth_conv_contrib = ConvDepthReductionBuffer(gpu, np.zeros(
                (int(np.ceil(last_s0/memory_sensitive_divisor)), self.in_img_pyr.levels[1], self.in_img_pyr.levels[1])))

            self.algorithm_conv_back = self.gpu.manager.algorithm(
                [buf_conv_prepool, self.depth_conv_err.buffer, self.depth_conv_contrib.buffer],
                spirv=shad_conv_back,
                workgroup=[int(np.ceil(last_s0 / memory_sensitive_divisor)), 0, 0],
                spec_consts=np.asarray([memory_sensitive_divisor, self.in_img_pyr.levels[1], last_s0], dtype=np.uint32).view(
                    np.float32)
            )

            self.conv_err_reductions = []
            prev_s0 = int(np.ceil(last_s0 / memory_sensitive_divisor))
            s0 = int(np.ceil(last_s0 / memory_sensitive_divisor))
            while True:
                if prev_s0 <= 1:
                    break
                self.conv_err_reductions.append(
                    self.gpu.manager.algorithm(
                        [self.depth_conv_err.buffer, self.depth_conv_contrib.buffer],
                        spirv=shad_conv_reduce,
                        workgroup=[s0, 0, 0],
                        spec_consts=np.asarray([memory_sensitive_divisor, self.in_img_pyr.levels[1], prev_s0], dtype=np.uint32).view(
                            np.float32)
                    )
                )
                # if s0 <= memory_sensitive_divisor:  # do while loop to ensure we include the final reduce
                #    break
                prev_s0 = s0
                s0 = int(np.ceil(s0 / memory_sensitive_divisor))

        self.basic_sequence = None  # mostly for debugging

    def forward_ops(self):
        return[
            kp.OpAlgoDispatch(self.forward_algorithm)
        ]

    def backward_ops(self):
        ops = []
        if self.backprop_input:
            ops.append(kp.OpAlgoDispatch(self.algorithm_input_back))
        if self.backprop_conv:
            ops.append(kp.OpAlgoDispatch(self.algorithm_conv_prepool_1))
            ops.extend(kp.OpAlgoDispatch(pr) for pr in self.conv_prepool_reductions)
            ops.append(kp.OpAlgoDispatch(self.algorithm_conv_back))
            ops.extend(kp.OpAlgoDispatch(ce) for ce in self.conv_err_reductions)
        return ops

    def optim_ops(self):
        return []

    def optim_buffs(self):
        # note: it's fine that depth_conv_err is larger than depth conv.
        #  Optim implementations should take depth_conv.size as input
        return [self.depth_conv, self.depth_conv_err, self.depth_conv_contrib, self.depth_conv_str]

    def basic_forward(self, image_pyr):
        # todo: make this a superclass method
        self.in_img_pyr.set(image_pyr.image_buffer.data())
        if self.basic_sequence is None:
            self.basic_sequence = self.gpu.manager.sequence()
            self.basic_sequence.record(kp.OpTensorSyncDevice([*self.forward_input_buffers]))
            for f in self.forward_ops():
                self.basic_sequence.record(f)
            self.basic_sequence.record(kp.OpTensorSyncLocal([*self.forward_output_buffers]))
        self.basic_sequence.eval()
        return self.out_pyr

    def display_basic_forward_sequence(self, image, display=None):
        if display is None:
            display = DirectDisplay()
        out_images = self.basic_forward(image)
        for i, o in enumerate(out_images.get()):
            display.imshow(f'output {i}', o)
        while True:
            display.update()
            if display.window.is_closing:
                break



if __name__ == '__main__':
    import pickle
    with open("../../../test/files/test_ai_pyr_pls_ignore.pyr", mode='rb') as f:

        gpu = GPUManager()
        im_pyr = pickle.load(f).to(gpu)

        pyr = DepthPyrConv(gpu, im_pyr, depth_edge_matrix)
        pyr.display_basic_forward_sequence(im_pyr)

    #im_pyr = pyr.run_basic_forward_sequence(im)
    #with open("test_files/test_ai_pyr_pls_ignore.pyr", mode='wb') as f:
    #    pickle.dump(im_pyr, f)

