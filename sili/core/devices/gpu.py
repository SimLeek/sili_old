import os
import subprocess
import sys

import kp


class GPUManager(object):
    """
    Manages a GPU device.

    Please keep all device names. The main benefit to this library over pytorch should be that it
      make devices and shader code *much* clearer, rather than abstracting it away. While that
      might make developing standard code harder, it should make developing alternate machine
      learning much easier.
    """

    def __init__(self, manager: kp.Manager = None):
        if manager is None:
            self.manager = kp.Manager()
        elif isinstance(manager, int):
            self.manager = kp.Manager(manager)
        else:
            self.manager = manager

        # save these important variables so we don't have to ping vulkan and the device constantly:
        self.max_workgroup_invocations = self.manager.get_device_properties()['max_work_group_invocations']
        # todo: get this through kompute or vulkan: https://github.com/KomputeProject/kompute/issues/360
        # this variable is specifically necessary for reductions ops often used in sparse shaders:
        self.maxComputeSharedMemorySize = 49152

    def buffer(self, data, type=None):
        """Returns an SSBO buffer. It's not a 'tensor'. (Try using np.float32 types with type)"""
        if type is None:
            return self.manager.tensor(data)
        else:
            return self.manager.tensor(data, tensor_type=type)


def get_shader(filename):
    if filename.endswith('.glsl') or filename.endswith('.comp'):
        spv_filename = filename[:-5] + '.spv'
        if not os.path.exists(spv_filename):
            try:
                subprocess.run(["glslc", filename, "-o", spv_filename],
                                        check=True,
                                        stdout=sys.stdout,
                                        stderr=sys.stderr, text=True)
            except subprocess.CalledProcessError as e:
                print("glslc command failed with output:")
                print(e.stdout)
                print(e.stderr, file=sys.stderr)
                raise e

        with open(spv_filename, 'rb') as f:
            shader = f.read()
    else:
        raise ValueError("Invalid file extension. Filename must end with .glsl or .comp")

    return shader
