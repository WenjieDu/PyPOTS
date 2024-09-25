"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
from torch import nn


class Splitting(nn.Module):
    def __init__(self):
        super().__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        """Returns the odd and even part"""
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(
        self,
        in_planes,
        splitting=True,
        kernel=5,
        dropout=0.5,
        groups=1,
        hidden_size=1,
        INN=True,
    ):
        super().__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1  # by default: stride==1
            pad_r = self.dilation * (self.kernel_size) // 2 + 1  # by default: stride==1

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1  # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(
                in_planes * prev_size,
                int(in_planes * size_hidden),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
                groups=self.groups,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                int(in_planes * size_hidden),
                in_planes,
                kernel_size=3,
                stride=1,
                groups=self.groups,
            ),
            nn.Tanh(),
        ]
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(
                in_planes * prev_size,
                int(in_planes * size_hidden),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
                groups=self.groups,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                int(in_planes * size_hidden),
                in_planes,
                kernel_size=3,
                stride=1,
                groups=self.groups,
            ),
            nn.Tanh(),
        ]

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(
                in_planes * prev_size,
                int(in_planes * size_hidden),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
                groups=self.groups,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                int(in_planes * size_hidden),
                in_planes,
                kernel_size=3,
                stride=1,
                groups=self.groups,
            ),
            nn.Tanh(),
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(
                in_planes * prev_size,
                int(in_planes * size_hidden),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
                groups=self.groups,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                int(in_planes * size_hidden),
                in_planes,
                kernel_size=3,
                stride=1,
                groups=self.groups,
            ),
            nn.Tanh(),
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))

            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)

            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups, hidden_size, INN):
        super().__init__()
        self.level = Interactor(
            in_planes=in_planes,
            splitting=True,
            kernel=kernel,
            dropout=dropout,
            groups=groups,
            hidden_size=hidden_size,
            INN=INN,
        )

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)


class LevelSCINet(nn.Module):
    def __init__(self, in_planes, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.interact = InteractorLevel(
            in_planes=in_planes,
            kernel=kernel_size,
            dropout=dropout,
            groups=groups,
            hidden_size=hidden_size,
            INN=INN,
        )

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)  # even: B, T, D odd: B, T, D


class SCINet_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.current_level = current_level

        self.workingblock = LevelSCINet(
            in_planes=in_planes,
            kernel_size=kernel_size,
            dropout=dropout,
            groups=groups,
            hidden_size=hidden_size,
            INN=INN,
        )

        if current_level != 0:
            self.SCINet_Tree_odd = SCINet_Tree(
                in_planes,
                current_level - 1,
                kernel_size,
                dropout,
                groups,
                hidden_size,
                INN,
            )
            self.SCINet_Tree_even = SCINet_Tree(
                in_planes,
                current_level - 1,
                kernel_size,
                dropout,
                groups,
                hidden_size,
                INN,
            )

    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)  # L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_, 0).permute(1, 0, 2)  # B, L, D

    def forward(self, x):
        x_even_update, x_odd_update = self.workingblock(x)
        # We recursively reordered these sub-series.
        # You can run the ./utils/recursive_demo.py to emulate this procedure.
        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))


class EncoderTree(nn.Module):
    def __init__(self, in_planes, num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.levels = num_levels
        self.SCINet_Tree = SCINet_Tree(
            in_planes=in_planes,
            current_level=num_levels - 1,
            kernel_size=kernel_size,
            dropout=dropout,
            groups=groups,
            hidden_size=hidden_size,
            INN=INN,
        )

    def forward(self, x):
        x = self.SCINet_Tree(x)
        return x
