
from .plugin import Plugin


class ConsoleBatchProgress(Plugin):
    def __init__(self, print_to_std_out=True):
        super(ConsoleBatchProgress, self).__init__()
        self._print_to_std_out = print_to_std_out
        self.n_batches = None
        self.ith_batch = None
        self.max_epochs = None
        self.current_epoch_number = None
        self.max_batches = None

        self._blanks = 50*' '

    def register(self, trainer):
        trainer.events.pre_epoch.append(self.pre_epoch_handler)
        trainer.events.post_batch.append(self.post_batch_handler)
        trainer.events.post_epoch.append(self.post_epoch_handler)

    def pre_epoch_handler(self, **kwargs):
        self.max_batches = len(kwargs['train_data'])
        self.max_epochs = kwargs['max_epochs']
        self.current_epoch_number = kwargs['current_epoch_number']

    def post_batch_handler(self, **kwargs):
        current_batch_number = kwargs['current_batch_number']
        print(self._blanks, end='\r')

        str = """Epoch {}/{} Batch {}/{} ({:.2f} %)""".format(self.current_epoch_number,
                                                                self.max_epochs,
                                                                current_batch_number,
                                                                self.max_batches,
                                                                100*current_batch_number/self.max_batches)

        print(str, end='\r')

    def post_epoch_handler(self, **kwargs):
         print('')
