from typing import Callable
import torch
import torch.multiprocessing as mp
import inspect
import itertools
import contextlib
import sys


class DummyFile(object):
    def write(self, x): pass

    def flush(self, *args, **kwargs): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


class _Task():
    def __init__(
            self,
            experiment_fn,
            device_manager,
            lock):

        self.fn = experiment_fn
        self.device_manager = device_manager
        self.lock = lock

    def __call__(self, args_kwargs):
        with self.lock:

            device_id = None
            min_counter = min(self.device_manager.values())

            # find one of the devices currently running the least number
            # of jobs and take it ...
            for k, v in self.device_manager.items():
                if v == min_counter:
                    device_id = k

                    break

            self.device_manager[device_id] += 1

        assert device_id is not None
        args, kwargs = args_kwargs

        try:
            with torch.cuda.device(device_id):
                with nostdout():
                    result = self.fn(*args, **kwargs)

        except Exception as ex:
            with self.lock:
                self.device_manager[device_id] -= 1

            return ex, args_kwargs

        else:
            with self.lock:
                self.device_manager[device_id] -= 1

            return result, args_kwargs


def scatter_fn_on_devices(
        fn: Callable,
        fn_args_kwargs: list,
        visible_devices: list,
        max_process_on_device: int):

    assert isinstance(fn_args_kwargs, list)
    assert isinstance(visible_devices, list)
    assert all((i < torch.cuda.device_count() for i in visible_devices))

    num_device = len(visible_devices)

    manager = mp.Manager()
    device_counter = manager.dict({t: 0 for t in visible_devices})
    lock = manager.Lock()

    task = _Task(
        fn,
        device_counter,
        lock)

    ret = []
    with mp.Pool(num_device*max_process_on_device, maxtasksperchild=1) as pool:

        for i, r in enumerate(pool.imap_unordered(task, fn_args_kwargs)):

            ret.append(r)
            result, args_kwargs = r

            if not isinstance(result, Exception):
                print("# Finished job {}/{}".format(
                    i + 1,
                    len(fn_args_kwargs)))

            else:
                print("#")
                print("# Error in job {}/{}".format(i, len(args_kwargs)))
                print("#")
                print("# Error:", type(result))
                print(repr(result))
                print("# experiment configuration:")
                print(args_kwargs)

    return ret


def keychain_value_iter(d, key_chain=None, allowed_values=None):
    key_chain = [] if key_chain is None else list(key_chain).copy()

    if not isinstance(d, dict):
        if allowed_values is not None:
            assert isinstance(d, allowed_values), 'Value needs to be of type {}!'.format(
                allowed_values)
        yield key_chain, d
    else:
        for k, v in d.items():
            yield from keychain_value_iter(
                v,
                key_chain + [k],
                allowed_values=allowed_values)


def configs_from_grid(grid):
    tmp = list(keychain_value_iter(grid, allowed_values=(list, tuple)))
    values = [x[1] for x in tmp]
    key_chains = [x[0] for x in tmp]

    ret = []

    for v in itertools.product(*values):

        ret_i = {}

        for kc, kc_v in zip(key_chains, v):
            tmp = ret_i
            for k in kc[:-1]:
                if k not in tmp:
                    tmp[k] = {}

                tmp = tmp[k]

            tmp[kc[-1]] = kc_v

        ret.append(ret_i)

    return ret
