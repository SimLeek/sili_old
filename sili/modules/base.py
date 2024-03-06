import abc

from sili.core.devices.gpu import GPUManager, get_shader


class Module(object):
    def __init__(self, gpu: GPUManager):
        self.gpu = gpu

        self.has_forward = False
        self.has_backward = False
        self.has_optim = False

        # for large constant arrays
        self.optim_setup_buffers = []
        self.forward_setup_buffers = []
        self.backward_setup_buffers = []

        # for input to the GPU.
        # REMEMBER: Do not take an image output from the same gpu and feed it back in here.
        #   Just leave that on the GPU for huge speedup thanks to not having unnecessary IO.
        self.optim_input_buffers = []
        self.forward_input_buffers = []
        self.backward_input_buffers = []

        # for output from the GPU
        #   You can take output from any buffer for debugging, but these are the outputs during normal operation.
        self.optim_output_buffers = []
        self.forward_output_buffers = []
        self.backward_output_buffers = []

    def setup_check(self):
        assert any([self.has_forward, self.has_backward, self.has_optim]), "Module must do *something*."
        assert any([len(self.optim_input_buffers) != 0,
                    len(self.forward_input_buffers) != 0,
                    len(self.backward_input_buffers) != 0]), \
            "Module must take input."
        assert any([len(self.optim_output_buffers) != 0,
                    len(self.forward_output_buffers) != 0,
                    len(self.backward_output_buffers) != 0]), \
            "Module must give output."

    @abc.abstractmethod
    def forward_ops(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def backward_ops(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def optim_ops(self):
        raise NotImplementedError()
