from .plugin import Plugin
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import OrderedDict
from torch.autograd import Variable


class PredictionMonitor(Plugin):
    def __init__(self, test_data: DataLoader, eval_every_n_epochs=1, print_to_std_out=False):
        super(PredictionMonitor, self).__init__()

        self.eval_ever_n_epochs = eval_every_n_epochs
        self._test_data = test_data
        self._print_to_std_out = print_to_std_out

        self.accuracies = OrderedDict()
        self.confusion_matrices = OrderedDict()

        self.evaluated_this_epoch = False

    def register(self, trainer):
        trainer.events.post_epoch.append(self.post_epoch_handler)

    def post_epoch_handler(self, **kwargs):
        epoch_count = kwargs['epoch_count']

        if epoch_count % self.eval_ever_n_epochs != 0:
            self.evaluated_this_epoch = False
            return

        else:
            model = kwargs['model']
            cuda = kwargs['cuda']
            model.eval()

            target_list = []
            predictions_list = []

            if self._print_to_std_out:
                print('testing...', end='\n')

            for batch_input, target in self._test_data:
                if cuda:
                    batch_input = batch_input.cuda()

                batch_input = Variable(batch_input)

                output = model(batch_input).data
                predictions = output.max(1)[1].type_as(target)

                target_list += target.view(-1).tolist()
                predictions_list += predictions.view(-1).tolist()

            self.accuracies[epoch_count] = accuracy_score(target_list, predictions_list)
            self.confusion_matrices[epoch_count] = confusion_matrix(target_list, predictions_list)
            self.evaluated_this_epoch = True

            if self._print_to_std_out:
                print(str(self))

    @staticmethod
    def last_of_orddict(ordered_dict):
        return ordered_dict[next(reversed(ordered_dict))]

    @property
    def accuracy(self):
        return self.last_of_orddict(self.accuracies)

    @property
    def confusion_matrix(self):
        return self.last_of_orddict(self.confusion_matrices)

    def __str__(self):
        if self.evaluated_this_epoch:
            return """Accuracy: {:.2f} %""".format(100*self.accuracy)
        else:
            return "Accuracy: na"


class LossMonitor(Plugin):
    def __init__(self):
        super(LossMonitor, self).__init__()
        self.losses_by_epoch = []
        self._losses_current_epoch = []

    def register(self, trainer):
        trainer.events.post_batch_backward.append(self.post_batch_backward_handler)
        trainer.events.post_epoch.append(self.post_epoch_handler)

    def post_batch_backward_handler(self, **kwargs):
        self._losses_current_epoch.append(kwargs['loss'])

    def post_epoch_handler(self, **kwargs):
        self.losses_by_epoch.append(self._losses_current_epoch)
        self._losses_current_epoch = []
