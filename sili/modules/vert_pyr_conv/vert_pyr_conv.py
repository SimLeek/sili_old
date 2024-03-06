import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os
from sili.modules.base import Module

from sili.core.buffers import ImageBuffer, ImagePyramidBuffer, calculate_pyramid_levels, ConvVertPyrBuffer, ConvDepthReductionBuffer
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

def calc_pyr_reduce_buf_size(levels, workgroup_size, channels = 3):
    num_levels = levels[1]
    levels_array = levels[2:]
    total_size = 0
    for i in range(num_levels):
        width = levels_array[3*i+1]
        height = levels_array[3*i+2]
        total_size += int(np.ceil(width*height*channels/workgroup_size))
    return total_size

class DepthPyrConv(Module):
    def __init__(self, gpu: GPUManager, image_pyr: ImagePyramidBuffer, init_conv=None, backprop_input_buf=None,
                 backprop_conv=False):
        """

        :param gpu: the GPUManager
        :param image_pyr: Either an ImagePyramidBuffer or a list containing the widths and heights of all images in pyr
        """
        super().__init__(gpu)

        self.forward_shader = get_shader(file_path + os.sep + 'vert_pyr_forward.comp')
        if not isinstance(image_pyr, ImagePyramidBuffer):
            self.in_img_pyr = ImagePyramidBuffer(self.gpu, image_pyr)
        else:
            self.in_img_pyr = image_pyr

        self.out_pyr = ImagePyramidBuffer(gpu, self.in_img_pyr.levels, use_lvl_buf=False)
        if callable(init_conv):
            self.vert_conv = ConvVertPyrBuffer(gpu, init_conv(self.in_img_pyr.levels[1]))
        elif isinstance(init_conv, np.ndarray):
            self.vert_conv = ConvVertPyrBuffer(gpu, init_conv)
        else:
            self.vert_conv = ConvVertPyrBuffer(gpu, np.zeros((self.in_img_pyr.levels[1], 3)))
        self.depth_conv_str = ConvVertPyrBuffer(gpu, np.zeros_like(self.vert_conv.buffer.data()))  # adagrad weights

        # PIPELINE OBJECTS:
        # FORWARD
        # these need to be accessible so that kompute can record input/output for pipelines
        self.forward_input_buffers = [self.in_img_pyr.image_buffer, self.in_img_pyr.pyr_lvl_buffer, self.vert_conv.buffer]
        self.forward_output_buffers = [self.out_pyr.image_buffer]

        self.forward_algorithm = self.gpu.manager.algorithm(
            [self.in_img_pyr.image_buffer, self.in_img_pyr.pyr_lvl_buffer, self.out_pyr.image_buffer, self.vert_conv.buffer],
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

            self.backward_input_buffers = [self.out_pyr.image_buffer, self.in_img_pyr.pyr_lvl_buffer, self.vert_conv.buffer, self.out_pyr_err]
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
            shad_conv_back_prepool_1 = get_shader(file_path + os.sep + 'depth_conv_back_prepool_full.comp')
            shad_conv_back_prepool_2 = get_shader(file_path + os.sep + 'index2_value2_sum_reduction.comp')
            shad_conv_back = get_shader(file_path + os.sep + 'index2_value2_to_conv_kernels.comp')

            # private buffers
            num_levels = self.in_img_pyr.levels[1]
            level_data = self.in_img_pyr.levels[2:]
            l_index = 0
            l_indices = []
            for l in range(num_levels):  # calculate the size of the output reduction... idk how tf...
                l_indices.append(l_index)  # needed for reduction step. Should go from 0,1000,1500,... to 0,1,2,...
                l_size = level_data[l*3+1]*level_data[l*3+2]*self.in_img_pyr.channels
                workgroup_size = min(l_size*num_levels,self.gpu.max_workgroup_invocations)
                l_index = l_index + int(np.ceil(l_size / workgroup_size) * num_levels)
            #l_index += num_levels  # last one

            self.buf_conv_prepool = self.gpu.manager.tensor(np.zeros([int(l_index*4)]))

            # no special backwards input/output buffers, because the backprop ends in the conv buffer inside this object
            self.algorithm_conv_prepools_1 = []
            l_index = 0
            for l in range(num_levels):
                #if l==0:
                #    continue # debugging bs indices
                l_start = level_data[l*3]*self.in_img_pyr.channels
                l_size = level_data[l*3+1]*level_data[l*3+2]*self.in_img_pyr.channels
                workgroup_size = min(l_size*num_levels,self.gpu.max_workgroup_invocations)
                self.algorithm_conv_prepools_1.append(self.gpu.manager.algorithm(
                    [self.in_img_pyr.image_buffer, self.out_pyr_err.image_buffer, self.buf_conv_prepool, self.in_img_pyr.pyr_lvl_buffer, self.vert_conv.buffer,
                     self.depth_conv_str.buffer],
                    spirv=shad_conv_back_prepool_1,
                    workgroup=[int(np.ceil(l_size*num_levels / workgroup_size)), 0, 0],
                    # workgroup=[1, 0, 0],
                    spec_consts=np.asarray([workgroup_size, l, l_index], dtype=np.uint32).view(np.float32)
                ))
                l_index = l_index + int(np.ceil(l_size/workgroup_size)*num_levels)

            self.conv_prepool_reductions = []
            while True:
                l_diffs = [l_indices[i+1]-l_indices[i] for i in range(len(l_indices)-1)]
                l_done = all([l==num_levels for l in l_diffs])
                if l_done:
                    break

                l_index = 0
                l_indices_2 = []
                l_sizes = []
                w_sizes = []
                for l in range(num_levels):  # calculate the size of the output reduction... idk how tf...
                    l_indices_2.append(l_index)  # needed for reduction step. Should go from 0,1000,1500,... to 0,1,2,...
                    l_size = l_diffs[l]/num_levels if l<len(l_indices)-1 else 1
                    l_sizes.append(l_size)
                    workgroup_size = min(l_size*num_levels, self.gpu.max_workgroup_invocations)
                    w_sizes.append(workgroup_size)
                    l_index = l_index + int(np.ceil(l_size / workgroup_size) * num_levels)
                for li,li2,ls,ws in zip(l_indices, l_indices_2, l_sizes, w_sizes):
                    self.conv_prepool_reductions.append(
                        self.gpu.manager.algorithm(
                            [self.buf_conv_prepool],
                            spirv=shad_conv_back_prepool_2,
                            workgroup=[int(np.ceil(ls*num_levels/ws)), 0, 0],
                            spec_consts=np.asarray([ws, int(ls*num_levels+li), li, li2, num_levels], dtype=np.uint32).view(np.float32)
                        )
                    )
                l_indices = l_indices_2

            self.depth_conv_err = ConvDepthBuffer(gpu, np.zeros(
                (self.in_img_pyr.levels[1], self.in_img_pyr.levels[1])))
            self.depth_conv_contrib = ConvDepthBuffer(gpu, np.zeros(
                (self.in_img_pyr.levels[1], self.in_img_pyr.levels[1])))

            # should also be its own function:
            num_levels = self.in_img_pyr.levels[1]
            levels_array = self.in_img_pyr.levels[2:]
            div_conv = np.ones((num_levels, num_levels))
            for i in range(num_levels):
                width1 = levels_array[3 * i + 1]
                height1 = levels_array[3 * i + 2]
                size1 = int(np.ceil(width1 * height1 * self.in_img_pyr.channels))
                for j in range(num_levels):
                    width2 = levels_array[3 * j + 1]
                    height2 = levels_array[3 * j + 2]
                    size2 = int(np.ceil(width2 * height2 * self.in_img_pyr.channels))

                    #div_conv[:, i] = size  # in*num_lvl+out
                    #div_conv[:, i] = np.sqrt(size)  # in*num_lvl+out
                    #div_conv[:, i] = 1
                    if size1>=size2:
                        div_conv[i, j] = size1/size2
                    else:
                        div_conv[i, j] = size2/size1




            self.depth_conv_div = ConvDepthBuffer(gpu, div_conv)
            self.backward_input_buffers.append(self.depth_conv_div.buffer)  # need for setup, otherwise this is all div0
            workgroup_size = min(self.depth_conv_err.size, self.gpu.max_workgroup_invocations)

            self.algorithm_conv_back = self.gpu.manager.algorithm(
                [self.buf_conv_prepool, self.depth_conv_err.buffer, self.depth_conv_contrib.buffer, self.depth_conv_div.buffer],
                spirv=shad_conv_back,
                workgroup=[int(np.ceil(self.depth_conv_err.size / workgroup_size)), 0, 0],
                spec_consts=np.asarray([workgroup_size, num_levels], dtype=np.uint32).view(
                    np.float32)
            )

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
            ops.extend(kp.OpAlgoDispatch(p1) for p1 in self.algorithm_conv_prepools_1)
            ops.extend(kp.OpAlgoDispatch(pr) for pr in self.conv_prepool_reductions)
            ops.append(kp.OpAlgoDispatch(self.algorithm_conv_back))
        return ops

    def optim_ops(self):
        return []

    def optim_buffs(self):
        # note: it's fine that depth_conv_err is larger than depth conv.
        #  Optim implementations should take depth_conv.size as input
        return [self.vert_conv, self.depth_conv_err, self.depth_conv_contrib, self.depth_conv_str]

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

