"""
ConvLSTM layer implementation, adapted from https://github.com/jhhuang96/ConvLSTM-PyTorch/blob/master/ConvRNN.py and
https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, k_size=3, padding=1, bias=False, norm=True):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch

        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_ch + self.hidden_ch, 4 * self.hidden_ch, kernel_size=k_size, padding=padding,
                          bias=bias),
                nn.GroupNorm(4 * self.hidden_ch // 32, 4 * self.hidden_ch))
        else:
            self.conv = nn.Conv2d(self.in_ch + self.hidden_ch, 4 * self.hidden_ch, kernel_size=k_size, padding=padding,
                                  bias=bias)

        self.hidden = None
        self._frozen = False

    def forward(self, x):
        if self.hidden is None:
            self._init_hidden(x, dims=list(range(x.shape[0])))
        h_curr, c_curr = self.hidden
        combined = torch.cat((x, h_curr), dim=1)
        combined_conv = self.conv(combined)

        ingate, forgetgate, cellgate, outgate = torch.split(combined_conv, self.hidden_ch, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c_next = forgetgate * c_curr + ingate * cellgate
        h_next = outgate * torch.tanh(c_next)

        if not self._frozen:
            self.hidden = h_next.clone().detach(), c_next.detach()

        return h_next

    def freeze(self):
        self._frozen = True

    def defrost(self):
        self._frozen = False

    def _reset(self):
        self.hidden = None

    def _init_hidden(self, x, dims):
        b, _, h, w = x.shape
        if self.hidden is not None and b != self.hidden[0].shape[0]:
            self.hidden = None
        if self.hidden is None:
            self.hidden = (torch.zeros(b, self.hidden_ch, h, w, device=x.device),
                           torch.zeros(b, self.hidden_ch, h, w, device=x.device))
        else:
            self.hidden[0][dims] = torch.zeros(len(dims), self.hidden_ch, h, w, device=x.device)
            self.hidden[1][dims] = torch.zeros(len(dims), self.hidden_ch, h, w, device=x.device)


class ConvLSTMLayer(nn.Module):
    def __init__(self, in_ch=256, hidden_ch=[], out_ch=None, k_size=3, padding=1, norm=True):
        super().__init__()

        assert isinstance(hidden_ch, list)
        if out_ch is None:
            out_ch = in_ch

        in_channels = [in_ch] + hidden_ch
        out_channels = hidden_ch + [out_ch]
        assert len(in_channels) == len(out_channels)

        if isinstance(k_size, tuple):
            assert len(k_size) == len(in_channels)
        if isinstance(padding, tuple):
            assert len(padding) == len(in_channels)
        if isinstance(norm, tuple):
            assert len(norm) == len(in_channels)

        self.layers = nn.ModuleList([
            ConvLSTMCell(in_ch=in_channels[l], hidden_ch=out_channels[l],
                         k_size=k_size[l] if isinstance(k_size, tuple) else k_size,
                         padding=padding[l] if isinstance(padding, tuple) else padding,
                         bias=False, norm=norm[l] if isinstance(norm, tuple) else norm) for l in range(len(in_channels))
        ])

        self.curr_data_str = None

    def forward(self, x, data_str=None):
        if data_str is not None:
            if self.curr_data_str is None:
                self.curr_data_str = data_str
                self._init_hidden(x, dims=torch.arange(data_str.shape[0]))
            else:
                # compute batch-wise differences
                diffs = torch.all(torch.eq(data_str, self.curr_data_str), dim=1)
                tm = torch.where(~diffs)
                if not torch.all(diffs):
                    self.curr_data_str = data_str
                    self._init_hidden(x, dims=torch.where(~diffs))

        for layer in self.layers:
            x = layer(x)
        return x

    def _init_hidden(self, x, dims=[]):
        b, _, h, w = x.shape
        for layer in self.layers:
            layer._init_hidden(x, dims=dims)

    def _reset(self):
        for layer in self.layers:
            layer._reset()

    def _freeze(self):
        for layer in self.layers:
            layer.freeze()

    def _defrost(self):
        for layer in self.layers:
            layer.defrost()

    def _freeze_hidden(self):
        if len(self.layers) > 1:
            for layer in self.layers[1:]:
                layer.freeze()
