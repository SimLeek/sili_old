from io import BytesIO

import kp
import numpy as np


def serialize_buffer(buf):
    data = buf.data()
    bytes_io = BytesIO()
    np.savez_compressed(bytes_io, data=data)
    bytes_io.seek(0)
    return bytes_io


def deserialize_buffer( buf):
    buffer = np.load(buf, allow_pickle=False)['data']
    return buffer
