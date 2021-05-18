"""
torch.utils.data.Sampler for recurrent datasets.
"""

from torch.utils.data import Sampler, Dataset, DataLoader
from random import shuffle


class RecurrentSampler(Sampler):
    """
    Recurrent sampler assumes a nested list of data, where items from sublists should not be merged with other sublists:
    root
    |--f1
    |  |--d1
    |  |--d2
    |  |--...
    |  |--dm
    |--f2
    |  |--d1
    |  |--d2
    |  |--...
    |  |--dm
    |--...
    |--fn

    In this case, root is the data root for the motion / semantic dataset (m==10, n=5,000), and (see data/{motion, semantic}_dataset.py for more info).

    The sampler will produce the following structure (m=2, batch_size=3]:
    - batch1: [fi_d1, fj_d1, fk_d1]
    - batch2: [fi_d2, fj_d2, fk_d2]
    - batch3: [fl_d1, ...]
    - batch4: [fl_d2, ...]
    - ...

    The folder indices(i, j, k, l, ...) are shuffled per default; shuffling the sublist items can be toggled. With all
    shuffled the results could be the following:
    - batch1: [fl_d2, fi_d2, fk_d1]
    - batch2: [fl_d1, fi_d1, fk_d2]
    - batch3: [fj_d1, ...]
    - batch4: [fj_d2, ...]
    - ...

    Run the corresponding test case in tests/test_recurrent_sampler.py for a visual example.
    """
    def __init__(self, data_source):
        """
        paths is a list of lists containing respective paths.
        each sublist has to be used sequentially.
        """
        self.paths, self.batch_size, self.do_shuffle = data_source

    def __iter__(self):
        # shuffle nested list
        shuffle(self.paths)
        if self.do_shuffle:
            for p in self.paths:
                shuffle(p)

        # manually create a flat list to iterate over
        flat_list = []
        for i in range(0, len(self.paths) - self.batch_size + 1, self.batch_size):
            for k in range(len(self.paths[0])):
                for b in range(self.batch_size):
                    flat_list.append(self.paths[i + b][k])

        return iter(flat_list)

    def __len__(self):
        flat_list = []
        for i in range(0, len(self.paths) - self.batch_size + 1, self.batch_size):
            for k in range(len(self.paths[0])):
                for b in range(self.batch_size):
                    flat_list.append(self.paths[i + b][k])

        return len(flat_list)


def get_ids_for_sampler(num_samples, recurrent_length):
    ids = list(range(num_samples))
    ids = [ids[i:i + recurrent_length] for i in range(0, len(ids), recurrent_length)]
    return ids[:-1]  # drop last
