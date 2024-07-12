import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Torch version:", torch.__version__)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())