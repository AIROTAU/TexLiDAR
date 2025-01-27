import torch

"""
this script check the device we are using
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print("Device name:", torch.cuda.get_device_properties('cuda').name)
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
print(f'torch version: {torch.__version__}')

