import os

import mlx.core as mx

# MOESTREAM_CPU=1 keeps tests off the GPU while another job owns the machine
if os.environ.get("MOESTREAM_CPU"):
    mx.set_default_device(mx.cpu)
