import numpy as np
from typing import Union, Sequence, List, Tuple
from numbers import Real


class ImagePyramidBuffer:
    def __init__(self, levels: List[int], image_buffer: np.ndarray, channels: int = 1):
        """
        Initialize an ImagePyramidBuffer.

        Args:
            levels (List[int]): List containing the start, width, and height of each image level.
            image_buffer (np.ndarray): Flattened array of image data.
            channels (int): Number of channels in the image. Default is 1.
        """
        self.levels = levels
        self.image_buffer = image_buffer
        self.channels = channels

    def get(self) -> List[np.ndarray]:
        """
        Retrieve the image pyramid as a list of arrays.

        Returns:
            List[np.ndarray]: List of image levels as arrays.
        """
        im_list = []
        flattened_array = self.image_buffer
        for l in range(0, len(self.levels), 3):
            start, width, height = int(self.levels[l]), int(self.levels[l + 1]), int(self.levels[l + 2])
            end_idx = int(start * self.channels + (width * height * self.channels))
            image = flattened_array[int(start * self.channels):end_idx].reshape((width, height, self.channels))
            im_list.append(image)
        return im_list


def generate_workgroup_arrays(
        pyramid_buffer: ImagePyramidBuffer,
        gpu_workgroup_size: int,
        desired_sparsity: Union[Sequence[Real], Real]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate workgroup arrays for GPU processing from an image pyramid buffer.

    Args:
        pyramid_buffer (ImagePyramidBuffer): Image pyramid buffer containing levels and image data.
        gpu_workgroup_size (int): Size of the GPU workgroup.
        desired_sparsity (Union[Sequence[Real], Real]): Desired sparsity value(s).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays, the first containing tuples of (global_id, k, out_id),
                                       and the second containing indices to access the first array.
    """
    levels = pyramid_buffer.levels

    if isinstance(desired_sparsity, Real):
        desired_sparsity = [desired_sparsity] * (len(levels) // 3)
    elif isinstance(desired_sparsity, Sequence):
        assert len(desired_sparsity) == int(
            len(levels) / 3), "Desired sparsity length must be equal to the number of levels."
    else:
        raise TypeError("desired_sparsity must be either a numeric type or a sequence.")

    array1 = []
    array2 = []

    workgroup_id = 0
    local_id = 0
    i = 0
    out_id_counter = 0
    num_global_id = levels[-3] + levels[-2] * levels[-1]  # last id + last width*height

    global_id = 0  # Initialize global_id

    while global_id < num_global_id:
        level_start = levels[i*3]
        width = levels[i*3 + 1]
        height = levels[i*3 + 2]
        level_end = level_start + width * height
        sparsity = desired_sparsity[i // 3]

        if (workgroup_id + 1) * gpu_workgroup_size >= level_end:
            global_id = workgroup_id * gpu_workgroup_size + local_id
            num_pixels = level_end - global_id
            k = int(num_pixels * sparsity)
            out_id = out_id_counter
            out_id_counter += k
            array1.append((global_id, k, out_id))
            if local_id == 0:
                array2.append(len(array1) - 1)
            i += 1
            if i>=len(levels) // 3:
                break
            local_id = int(levels[i*3] % gpu_workgroup_size)
        else:
            num_pixels = gpu_workgroup_size
            k = int(num_pixels * sparsity)
            out_id = out_id_counter
            out_id_counter += k
            global_id = workgroup_id * gpu_workgroup_size + local_id
            array1.append((global_id, k, out_id))
            if local_id == 0:
                array2.append(len(array1) - 1)
            workgroup_id += 1
            local_id = 0
            if global_id>=num_global_id:
                break

    return np.array(array1), np.array(array2)


# Test Cases
def test_generate_workgroup_arrays():
    """
    Test the generate_workgroup_arrays function with different inputs and print the results.
    """
    levels1 = [0, 6, 8, 48, 64, 48]
    image_buffer1 = np.random.rand(48 * 48).flatten()
    pyramid_buffer1 = ImagePyramidBuffer(levels1, image_buffer1)

    gpu_workgroup_size1 = 64
    desired_sparsity1 = [0.5, 0.25]

    array1, array2 = generate_workgroup_arrays(pyramid_buffer1, gpu_workgroup_size1, desired_sparsity1)

    print("Test Case 1")
    print("Array1:", array1)
    print("Array2:", array2)
    print()

    levels2 = [0, 64, 48]
    image_buffer2 = np.random.rand(64 * 48).flatten()
    pyramid_buffer2 = ImagePyramidBuffer(levels2, image_buffer2)

    gpu_workgroup_size2 = 32
    desired_sparsity2 = 0.1

    array1, array2 = generate_workgroup_arrays(pyramid_buffer2, gpu_workgroup_size2, desired_sparsity2)

    print("Test Case 2")
    print("Array1:", array1)
    print("Array2:", array2)


def calculate_sparsities(pyramid_buffer: ImagePyramidBuffer, base_sparsity: Real) -> List[Real]:
    """
    Calculate the sparsities for each level of the image pyramid buffer.

    Args:
        pyramid_buffer (ImagePyramidBuffer): Image pyramid buffer containing levels and image data.
        base_sparsity (Real): The base sparsity value for the first level.

    Returns:
        List[Real]: List of sparsities for each level.
    """
    levels = pyramid_buffer.levels
    first_level_resolution = levels[1] * levels[2]  # Width * Height of the first level

    sparsities = []
    for i in range(0, len(levels), 3):
        current_level_resolution = levels[i + 1] * levels[i + 2]  # Width * Height of the current level
        sparsity = min(1.0, (first_level_resolution / current_level_resolution) * base_sparsity)
        sparsities.append(sparsity)

    return sparsities


# Test the function
def test_calculate_sparsities():
    """
    Test the calculate_sparsities function with different inputs and print the results.
    """
    levels = [0, 64, 48, 5076, 6, 8, 5192, 2, 1]
    image_buffer = np.random.rand(48 * 48).flatten()
    pyramid_buffer = ImagePyramidBuffer(levels, image_buffer)

    base_sparsity = 0.01
    sparsities = calculate_sparsities(pyramid_buffer, base_sparsity)

    print("Calculated Sparsities:", sparsities)

def calculate_sparsity_array_for_image_pyramid(L: int, sp_scale: float, img_scale: float, desired_sparsity: float) -> List[float]:
    """
    Calculate an array of sparsities for an image pyramid given the number of levels,
    sparsity scaling factor, image scaling factor, and desired total sparsity.

    Parameters:
    - L (int): Number of levels in the pyramid.
    - sp_scale (float): Sparsity scaling factor (beta).
    - img_scale (float): Image scaling factor (alpha).
    - desired_sparsity (float): Desired total sparsity (S).

    Returns:
    - List[float]: Array of sparsities for each level in the pyramid.
    """

    # Calculate n_c
    b = sp_scale
    S = desired_sparsity

    n_c = np.ceil(np.log(S) * -np.log(b))

    # Calculate s_0
    a = img_scale
    numerator = (a-b)*(a**n_c-a**L+a*S-S)
    denominator = (a-1)*b*((b/a)**n_c-1)
    s_0 = numerator / denominator

    # Generate sparsities array
    sparsities = []
    for i in range(L):
        sparsity = s_0 * (b ** i)
        if sparsity >= 1.0:
            sparsities.append(1.0)
        else:
            sparsities.append(sparsity)

    return sparsities

def test_calculate_sparsity_array_for_image_pyramid():
    # clearly doesn't actually work
    L = 12
    sp_scale = 2
    img_scale = 0.5
    desired_sparsity = 0.02

    sparsities = calculate_sparsity_array_for_image_pyramid(L, sp_scale, img_scale, desired_sparsity)
    print(sparsities)
    print(f"actual sparsity vs original image: {sum([x*img_scale**(i) for i,x in enumerate(sparsities)])*sum([img_scale**i for i in range(len(sparsities))])}")
    print(f"actual sparsity vs image pyramid: {sum([x*img_scale**(i) for i,x in enumerate(sparsities)])}")

# Run the test cases
# test_generate_workgroup_arrays()
#test_calculate_sparsities()
test_calculate_sparsity_array_for_image_pyramid()
