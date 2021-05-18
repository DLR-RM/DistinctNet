"""
Test (re-) initialization of hidden states in a ConvLSTM layer.
"""

from unittest import TestCase
import torch
from networks.convlstm import ConvLSTMLayer
from utils.train_utils import str_to_tens


class TestConvLSTMLayer(TestCase):
    def test_init(self):
        convlstm = ConvLSTMLayer(in_ch=256, out_ch=256, k_size=1, padding=0, norm=False).cuda()
        tens = torch.randn(2, 256, 15, 20).cuda()
        string = str_to_tens('foo.bar').unsqueeze(0).cuda().repeat(2, 1)

        for layer in convlstm.layers:
            self.assertIs(layer.hidden, None)
        _ = convlstm(tens, data_str=string)
        self.assertTrue(torch.all(torch.eq(string, convlstm.curr_data_str)))

    def test_double_forward_same(self):
        convlstm = ConvLSTMLayer(in_ch=256, out_ch=256, k_size=1, padding=0, norm=False).cuda()
        tens = torch.randn(2, 256, 15, 20).cuda()
        string = str_to_tens('foo.bar').unsqueeze(0).cuda().repeat(2, 1)

        _ = convlstm(tens, data_str=string)
        self.assertTrue(torch.all(torch.eq(string, convlstm.curr_data_str)))

        hidden_first = convlstm.layers[0].hidden
        hidden_first = [h.clone() for h in hidden_first]

        # second forward
        # the first batch is the same, while the second changed
        # we freeze so that we can compare the internal states to the previous pass
        convlstm._freeze()
        _ = convlstm(tens, data_str=string)
        self.assertTrue(torch.all(torch.eq(string, convlstm.curr_data_str)))

        hidden_second = convlstm.layers[0].hidden
        a, b = hidden_first
        c, d = hidden_second

        # hidden states should be different and zero for the second pass
        self.assertTrue(torch.all(torch.eq(a, c)))
        self.assertTrue(torch.all(torch.eq(b, d)))
        self.assertFalse(torch.sum(c) == 0)
        self.assertFalse(torch.sum(d) == 0)

    def test_double_forward_different(self):
        convlstm = ConvLSTMLayer(in_ch=256, out_ch=256, k_size=1, padding=0, norm=False).cuda()
        tens = torch.randn(2, 256, 15, 20).cuda()
        string = str_to_tens('foo.bar').unsqueeze(0).cuda().repeat(2, 1)

        _ = convlstm(tens, data_str=string)
        self.assertTrue(torch.all(torch.eq(string, convlstm.curr_data_str)))

        hidden_first = convlstm.layers[0].hidden
        hidden_first = [h.clone() for h in hidden_first]

        # second forward
        # the first batch is the same, while the second changed
        string = str_to_tens('bar.foo').unsqueeze(0).cuda().repeat(2, 1)
        # we freeze so that we can compare the internal states to the previous pass
        convlstm._freeze()
        _ = convlstm(tens, data_str=string)
        self.assertTrue(torch.all(torch.eq(string, convlstm.curr_data_str)))

        hidden_second = convlstm.layers[0].hidden
        a, b = hidden_first
        c, d = hidden_second

        # hidden states should be different and zero for the second pass
        self.assertFalse(torch.all(torch.eq(a, c)))
        self.assertFalse(torch.all(torch.eq(b, d)))
        self.assertTrue(torch.sum(c) == 0)
        self.assertTrue(torch.sum(d) == 0)

    def test_double_forward_mixed(self):
        convlstm = ConvLSTMLayer(in_ch=256, out_ch=256, k_size=1, padding=0, norm=False).cuda()
        tens = torch.randn(2, 256, 15, 20).cuda()
        string = str_to_tens('foo.bar').unsqueeze(0).cuda().repeat(2, 1)

        _ = convlstm(tens, data_str=string)
        self.assertTrue(torch.all(torch.eq(string, convlstm.curr_data_str)))

        hidden_first = convlstm.layers[0].hidden
        hidden_first = [h.clone() for h in hidden_first]

        # second forward
        # the first batch is the same, while the second changed
        st1 = str_to_tens('foo.bar').unsqueeze(0).cuda()
        st2 = str_to_tens('bar.foo').unsqueeze(0).cuda()
        string = torch.cat((st1, st2), dim=0)
        # we freeze so that we can compare the internal states to the previous pass
        convlstm._freeze()
        _ = convlstm(tens, data_str=string)
        self.assertTrue(torch.all(torch.eq(string, convlstm.curr_data_str)))

        hidden_second = convlstm.layers[0].hidden
        a, b = hidden_first
        c, d = hidden_second

        # the first batch states should be the same, while the second should be different and zero for the second pass
        self.assertTrue(torch.all(torch.eq(a[0], c[0])))
        self.assertTrue(torch.all(torch.eq(b[0], d[0])))
        self.assertFalse(torch.all(torch.eq(a[1], c[1])))
        self.assertFalse(torch.all(torch.eq(b[1], d[1])))
        self.assertTrue(torch.sum(c[1]) == 0)
        self.assertTrue(torch.sum(d[1]) == 0)
