class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def labels(self):
        raise NotImplementedError

    @property
    def sample_labels(self):
        raise NotImplementedError
