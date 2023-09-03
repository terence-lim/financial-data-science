"""Miscellaneous utilities

Copyright 2022, Terence Lim

MIT License
"""
import torch

def check_cuda():
    """Print diagnostics of cuda environment

    Notes:

    - https://pytorch.org/get-started/locally/
    - check cuda version (e.g. 11.4?): Nvidia-smi
    - install matching torch version

      - pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
      - you can specify $ export TORCH\_CUDA\_ARCH\_LIST="8.6" 
        in your environment to force it to build with SM 8.6 support
    """
    print('available', torch.cuda.is_available())
    print('version', torch.__version__)
    print('sm86 for rtx3080', torch.cuda.get_arch_list())
