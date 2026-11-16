"""Model classes that tsfast does not ship: TDNN, TCN with configurable receptive field, and
the left/right mirror-symmetry wrapper.

RNN/GRU/LSTM come from ``tsfast.models.SimpleRNN``, used unchanged. Normalisation is done
outside the model by tsfast's ``ScaledModel``.
"""
from torch import Tensor, nn
from torch.nn import Mish
from torch.nn.utils.parametrizations import weight_norm
from tsfast.models import CConv1D, CausalConv1d, SeqLinear  # package-level import: tsfast 0.4 moved the submodules


class TDNN(nn.Module):
    """Time-delay neural network (Waibel 1989): a causal tapped delay line (one causal
    convolution spanning ``delay + 1`` samples) feeding a time-shared MLP head."""

    def __init__(self, input_size: int, output_size: int, delay: int = 10,
                 hl_depth: int = 1, hl_width: int = 100, act=Mish, wn: bool = True, bn: bool = False):
        super().__init__()
        self.delay = delay
        self.tdl = CConv1D(input_size, hl_width, kernel_size=delay + 1, activation=act, wn=wn, bn=bn)
        self.head = SeqLinear(hl_width, output_size, hidden_size=hl_width, hidden_layer=hl_depth, act=act)

    def forward(self, x: Tensor) -> Tensor:
        out = self.tdl(x.transpose(1, 2))      # (B, T, C) -> (B, hl_width, T)
        return self.head(out.transpose(1, 2))  # -> (B, T, output_size)


class _TCNBlock(nn.Module):
    """Bai et al. (2018) residual block: ``convs_per_block`` dilated causal convolutions."""

    def __init__(self, input_size, output_size, kernel_size=3, dilation=1,
                 convs_per_block=2, activation=Mish, wn=True):
        super().__init__()
        layers = []
        for i in range(convs_per_block):
            conv = CausalConv1d(input_size if i == 0 else output_size, output_size,
                                kernel_size, dilation=dilation)
            if wn:
                conv = weight_norm(conv)
            layers += [conv, activation()]
        self.layers = nn.Sequential(*layers)
        self.residual = (nn.Conv1d(input_size, output_size, kernel_size=1)
                         if input_size != output_size else None)

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        return out + (x if self.residual is None else self.residual(x))


class TCN(nn.Module):
    """Causal TCN; receptive field = 1 + convs_per_block * (kernel_size - 1) * (2**hl_depth - 1)."""

    def __init__(self, input_size, output_size, hl_depth=1, hl_width=10,
                 kernel_size=3, convs_per_block=2, act=Mish, wn=True):
        super().__init__()
        self.conv_layers = nn.Sequential(*[
            _TCNBlock(input_size if i == 0 else hl_width, hl_width, kernel_size=kernel_size,
                      dilation=2 ** i, convs_per_block=convs_per_block, activation=act, wn=wn)
            for i in range(hl_depth)
        ])
        self.final = nn.Conv1d(hl_width, output_size, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.final(self.conv_layers(x.transpose(1, 2))).transpose(1, 2)


# Mirror of a driving manoeuvre about the vehicle's longitudinal axis: lateral channels change
# sign, left and right wheel speeds swap places.
NEGATE = ("acc_y_CAN_ms2", "yaw_rate_CAN_rads", "delta_sw_rad")
SWAP = (("wheel_vel_fl_CAN_ms", "wheel_vel_fr_CAN_ms"), ("wheel_vel_rl_CAN_ms", "wheel_vel_rr_CAN_ms"))


class SignAntisymmetricWrapper(nn.Module):
    """Built-in mirror symmetry: out(x) = (f(x) - f(Sx)) / 2, so beta(Sx) = -beta(x) exactly.

    S negates a_y, yaw rate and steering angle (mirrored about 0) and swaps the left/right wheel
    speeds of the input set ``u_names``. It acts on the raw physical channels, so it wraps the
    normalising ``ScaledModel`` from the outside, and beta must leave the model unnormalised.
    """

    def __init__(self, inner: nn.Module, u_names: list[str]):
        super().__init__()
        self.inner = inner  # not ``module``: tsfast's _unwrap_ddp strips any .module chain
        self.negate_idx = [u_names.index(n) for n in NEGATE if n in u_names]
        self.swap_pairs = [(u_names.index(a), u_names.index(b)) for a, b in SWAP if a in u_names and b in u_names]
        if not self.negate_idx:
            raise ValueError(f"mirror symmetry needs a lateral channel, inputs are {u_names}")

    def _mirror(self, x: Tensor) -> Tensor:
        x = x.clone()
        x[:, :, self.negate_idx] = -x[:, :, self.negate_idx]
        for i, j in self.swap_pairs:
            xi = x[:, :, i].clone()
            x[:, :, i] = x[:, :, j]
            x[:, :, j] = xi
        return x

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (self.inner(x) - self.inner(self._mirror(x)))
