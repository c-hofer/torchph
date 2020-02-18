from typing import Callable
import torch
import multiprocessing as mp
import itertools
import contextlib
import sys
import traceback
import time


from pathlib import Path


class DummyFile(object):
    def write(self, x): pass

    def flush(self, *args, **kwargs): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


class _ErrorCatchingProcess(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class _GPUTask():
    def __init__(
            self,
            fn,
            args_kwargs,
            gpu_id):

        self.fn = fn
        self.args_kwargs = args_kwargs
        self.gpu_id = gpu_id

    def __call__(self):
        with torch.cuda.device(self.gpu_id):
            with nostdout():
                args, kwargs = self.args_kwargs
                self.fn(*args, **kwargs)


class Distributor:

    def __init__(self,
                 fn: Callable,
                 args_kwargs: list,
                 visible_devices: list,
                 num_max_jobs_on_device: int,
                 log_file_path=None):

        self.visible_devices = visible_devices
        self.counter = {k: 0 for k in visible_devices}

        self.fn = fn
        self.args_kwargs = args_kwargs
        self.num_max_jobs = num_max_jobs_on_device
        self.log_file_path = None if log_file_path is None else Path(
            log_file_path)

        if self.log_file_path is not None:
            assert self.log_file_path.parent.is_dir()

        self.progress_wheel = itertools.cycle(['|', '/', '-', '\\'])

    def _log(self, *args):
        print(*args)

        if self.log_file_path is not None:
            with open(self.log_file_path, 'w') as fid:
                print(*args, file=fid)

    def _iter_free_devices(self):
        for k, v in self.counter.items():
            if v < self.num_max_jobs:
                yield k

    def run(self):

        q = list(enumerate(self.args_kwargs))

        processes = []

        finished_jobs = 0
        num_jobs = len(q)

        error_args_kwargs = []

        s = '=== Starting {} jobs on devices {} with {} jobs per device ==='
        self._log(s.format(
            num_jobs,
            ', '.join((str(i) for i in self.visible_devices)),
            self.num_max_jobs))

        while True:

            for gpu_id, (id, args_kwargs) in zip(self._iter_free_devices(), q):

                task = _GPUTask(self.fn, args_kwargs, gpu_id)
                proc = _ErrorCatchingProcess(group=None, target=task)
                proc.gpu_id = gpu_id
                proc.args_kwargs_id = id

                self.counter[gpu_id] += 1
                q.remove((id, args_kwargs))
                processes.append(proc)

                proc.start()
                time.sleep(1.0) # to make sure startup phases of process do not
                # interleave

            for p in processes:

                if p.exitcode is not None:
                    processes.remove(p)
                    self.counter[p.gpu_id] -= 1
                    finished_jobs += 1
                    print('Finished job {}/{}'.format(finished_jobs, num_jobs))

                    if p.exception is not None:
                        self._log('=== ERROR (job {}) ==='.format(
                            p.args_kwargs_id))
                        self._log(p.exception)
                        self._log('=============')
                        error_args_kwargs.append(
                            (p.exception, p.args_kwargs_id))

            if len(processes) == 0:
                break

            time.sleep(0.25)
            print(next(self.progress_wheel), end='\r')

        return error_args_kwargs


def scatter_fn_on_devices(
        fn: Callable,
        fn_args_kwargs: list,
        visible_devices: list,
        max_process_on_device: int):

    assert isinstance(fn_args_kwargs, list)
    assert isinstance(visible_devices, list)
    assert all((i < torch.cuda.device_count() for i in visible_devices))

    d = Distributor(
        fn,
        fn_args_kwargs,
        visible_devices,
        max_process_on_device
    )

    return d.run()


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
