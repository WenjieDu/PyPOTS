"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch


# overwrite autocast to make it compatible with both torch >=2.4 and <2.4
def autocast(**kwargs):
    if torch.__version__ < "2.4":
        from torch.cuda.amp import autocast

        return autocast(**kwargs)
    else:
        from torch.amp import autocast

        return autocast("cuda", **kwargs)
