import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os

from sili.core.buffers import ImageBuffer, ImagePyramidBuffer, calculate_pyramid_levels
from sili.core.devices.gpu import GPUManager, get_shader
from sili.modules.base import Module

file_path = os.path.dirname(os.path.abspath(__file__))


class ToImagePyramid(Module):
    def __init__(self,
                 gpu: GPUManager,
                 image: ImageBuffer,
                 scale: float = np.sqrt(2)):
        assert scale > 1.0, "Scale must be large enough to divide the image"

        self.reverse = False  # set this to true for pipeline creators to only use it for backwards

        self.gpu = gpu
        self.forward_shader = get_shader(file_path + os.sep + 'pyramid_gen.comp')
        if not isinstance(image, ImageBuffer):
            self.image = ImageBuffer(self.gpu, image)
        else:
            self.image = image
        pyr_levels = calculate_pyramid_levels(
            self.image.height,
            self.image.width,
            scale,
            self.image.colors
        )
        # self.geo_sum = 1 / (1 - (1 / (scale ** 2)))

        self.out_pyr = ImagePyramidBuffer(gpu, pyr_levels)

        self.has_forward = True
        self.has_backward = False
        self.has_optim = False

        # PIPELINE OBJECTS:
        # these need to be accessible so that kompute can record input/output for pipelines
        self.forward_input_buffers = [self.image.buffer, self.out_pyr.pyr_lvl_buffer]
        self.forward_output_buffers = [self.out_pyr.image_buffer]

        # todo: this may be a more general superclass method
        # this is the part that gets shoved into popelines
        self.forward_algorithm = self.gpu.manager.algorithm(
            [*self.forward_input_buffers, *self.forward_output_buffers],
            spirv=self.forward_shader,
            workgroup=[int(np.ceil(self.image.size / self.gpu.max_workgroup_invocations)), 0, 0],
            spec_consts=np.asarray([self.gpu.max_workgroup_invocations], dtype=np.uint32).view(np.float32)
        )

        self.basic_sequence = None  # mostly for debugging

    def forward_ops(self):
        return [
            kp.OpAlgoDispatch(self.forward_algorithm)
        ]

    def backward_ops(self):
        return []

    def optim_ops(self):
        return []

    def basic_forward(self, image):
        # todo: make this a superclass method
        self.image.set(image)
        if self.basic_sequence is None:
            self.basic_sequence = self.gpu.manager.sequence()
            self.basic_sequence.record(kp.OpTensorSyncDevice([*self.forward_input_buffers]))
            self.basic_sequence.record(kp.OpAlgoDispatch(self.forward_algorithm))
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


class ToImagePyramidReverse(ToImagePyramid):

    def forward_ops(self):
        return []

    def backward_ops(self):
        return super(self).forward_ops()


if __name__ == '__main__':
    gpu = GPUManager()
    im = cv2.imread("../../../test/files/test_ai_pls_ignore.png").astype(np.float32) / 255
    pyr = ToImagePyramid(gpu, im)
    # pyr.display_basic_forward_sequence(im)

    import pickle

    im_pyr = pyr.basic_forward(im)
    with open("../../../test/files/test_ai_pyr_pls_ignore.pyr", mode='wb') as f:
        pickle.dump(im_pyr, f)
    # with open("test_files/test_ai_pyr_pls_ignore.pyr", mode='rb') as f:
    #    im_pyr = pickle.load(f)
    #    print(im_pyr)
