import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os

from sili.core.buffers import ImageBuffer, ImagePyramidBuffer, calculate_pyramid_levels, ConvDepthBuffer, \
    ConvDepthReductionBuffer
from sili.core.devices.gpu import GPUManager, get_shader

file_path = os.path.dirname(os.path.abspath(__file__))


class Reduction(object):
    def __init__(self,
                 gpu: GPUManager,
                 reduction_io: kp.Tensor,
                 initial_size: int = None,  # for when we're only counting indices 0 to initial_size (pre-reduced)
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
        if not isinstance(reduction_io, kp.Tensor):
            raise TypeError("Prediction Buffer must be a buffer attached to a device")
        else:
            self.red_buff = reduction_io

        self.has_forward = forward
        if forward:
            self.forward_shader = get_shader(file_path + os.sep + 'reduction.comp')

            # PIPELINE OBJECTS:
            # FORWARD
            # these need to be accessible so that kompute can record input/output for pipelines
            self.forward_input_buffers = [self.red_buff]
            self.forward_output_buffers = [self.red_buff]

            # Note: I'm not going to sum the error into one number. It's kinda useful, but it costs a bit and you can do
            # that with a reduction op anywhere. Leaving it like this gives us options for nice error images, while
            # leaving options for making error graphs.
            # Note2: Currently, this reduces to num_workgroups in one step, which is a fair amount of reduction.
            # todo: Either make this return a full image, or do a full reduction to one number.
            self.forward_algorithms = []
            if initial_size is not None:
                reduced_size = initial_size
            else:
                reduced_size = int(self.red_buff.size())
            while reduced_size > 1:
                self.forward_algorithms.append(self.gpu.manager.algorithm(
                    [self.red_buff],
                    spirv=self.forward_shader,
                    workgroup=[int(np.ceil(reduced_size / min(reduced_size, self.gpu.max_workgroup_invocations))), 0,
                               0],
                    spec_consts=np.asarray([min(reduced_size, self.gpu.max_workgroup_invocations), reduced_size],
                                           dtype=np.uint32).view(np.float32)
                ))
                reduced_size = int(reduced_size / self.gpu.max_workgroup_invocations)

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
                self.basic_sequence.record(kp.OpAlgoDispatch(f))
            self.basic_sequence.record(kp.OpTensorSyncLocal([*self.forward_output_buffers]))
        self.basic_sequence.eval()
        return self.red_buff

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


if __name__ == '__main__':
    im = cv2.imread("../../../test/files/test_ai_pls_ignore.png").astype(np.float32) / 255
    gpu = GPUManager()
    red_buff = ImageBuffer(gpu, im)
    err = Reduction(gpu, red_buff.buffer)
    err.display_basic_forward_sequence(shape=im.shape)

    # im_pyr = pyr.run_basic_forward_sequence(im)
    # with open("test_files/test_ai_pyr_pls_ignore.pyr", mode='wb') as f:
    #    pickle.dump(im_pyr, f)
