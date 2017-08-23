class MappingIterable(object):
    def __init__(self, decoratee):
        if not hasattr(decoratee, '__iter__'):
            raise ValueError('decoratee does not implement __iter__')

        self._decoratee = decoratee

        def __getattr__(self, item):
            print(item)
            if item != '__iter__':
                return self._decoratee.__getattr__(item)