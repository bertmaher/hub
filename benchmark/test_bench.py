"""test_bench.py
Runs hub models in benchmark mode using pytest-benchmark. Run setup separately first.

Usage:
  python test.py --setup_only
  pytest test_bench.py

See pytest-benchmark help (pytest test_bench.py -h) for additional options
e.g. --benchmark-autosave
     --benchmark-compare
     -k <filter expression>
     ...
"""
import os
import pytest
import torch
from bench_utils import workdir, list_model_paths

def pytest_generate_tests(metafunc, display_len=24):
    # This is where the list of models to test can be configured
    # e.g. by using info in metafunc.config
    all_models = list_model_paths()
    short_names = []
    for name in all_models:
        short = os.path.split(name)[1]
        if len(short) > display_len:
           short = short[:display_len] + "..."
        short_names.append(short)
    metafunc.parametrize('model_path', all_models, ids=short_names, scope="module")
    metafunc.parametrize('device', ['cpu', 'cuda'], scope='module')
    metafunc.parametrize('jit', [True, False], ids=["script", "eager"], scope='module')
    metafunc.parametrize("executor_and_fuser", ["legacy-old", "profiling-te"], scope='module')

def set_fuser(fuser_name):
    if fuser_name == 'te':
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(True)
    elif fuser_name == 'old':
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser_name == 'none':
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)

def set_executor(executor_name):
    if executor_name == 'profiling':
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_bailout_depth(20)
    elif executor_name == 'simple':
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(False)
    elif executor_name == 'legacy':
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

@pytest.fixture(scope='module')
def hub_model(request, model_path, device, jit, executor_and_fuser):
    """Constructs a model object for pytests to use.
    Any pytest function that consumes a 'modeldef' arg will invoke this
    automatically, and reuse it for each test that takes that combination
    of arguments within the module.

    If reusing the module between tests isn't safe, change 'scope' parameter.
    """
    install_file = 'install.py'
    hubconf_file = 'hubconf.py'
    executor, fuser = executor_and_fuser.split("-")
    set_executor(executor)
    set_fuser(fuser)
    with workdir(model_path):
        hub_module = torch.hub.import_module(hubconf_file, hubconf_file)
        Model = getattr(hub_module, 'Model', None)
        if not Model:
            raise RuntimeError('Missing class Model in {}/hubconf.py'.format(model_path))
        return Model(device=device, jit=jit)

def cuda_sync(func, device, *args, **kwargs):
    func(*args, **kwargs)
    if device == 'cuda':
        torch.cuda.synchronize()

@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=True,
    group='hub',
)
class TestBenchNetwork:
    """
    This test class will get instantiated once for each 'model_stuff' provided
    by the fixture above, for each device listed in the device parameter.
    """
    def test_train(self, hub_model, device, benchmark):
        try:
            benchmark(cuda_sync, hub_model.train, device)
        except NotImplementedError:
            print('Method train is not implemented, skipping...')

    def test_eval(self, hub_model, device, benchmark):
        try:
            benchmark(cuda_sync, hub_model.eval, device)
        except NotImplementedError:
            print('Method eval is not implemented, skipping...')
