class ConditionalIterable:
    def __init__(self, iterable, pred):
        self.decoratee = iterable
        self.pred = pred

    def __iter__(self):
        return (x for x in self.decoratee if self.pred(x))

    def __len__(self):
        return len(self.decoratee)


class TransformingIterable:
    def __init__(self, iterable, transform):
        self.decoratee = iterable
        self.transform = transform

    def __iter__(self):
        return (self.transform(x) for x in self.decoratee)

    def __len__(self):
        return len(self.decoratee)
