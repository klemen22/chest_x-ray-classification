import torch
import os
import subprocess

print("CUDA IS AVAILABLE:", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
