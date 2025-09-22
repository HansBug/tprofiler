import pytest
import torch
import torch.multiprocessing as mp
from hbutils.system import get_free_port

from tprofiler.dist import gather
from .worker_context import spawn_worker_context, DistDataSaver


def gather_default_worker(rank, world_size, port, saver: DistDataSaver):
    with spawn_worker_context(rank, world_size, port):
        data = torch.randn(4 - rank, 3)
        gdata = gather(data)
        jdata = {
            'rank': rank,
            'data': data,
            'gdata': gdata,
        }
        saver.save_data(rank, jdata)


def gather_default_dim_worker(rank, world_size, port, saver: DistDataSaver):
    with spawn_worker_context(rank, world_size, port):
        data = torch.randn(3, 4 - rank)
        gdata = gather(data, dim=-1)
        jdata = {
            'rank': rank,
            'data': data,
            'gdata': gdata,
        }
        saver.save_data(rank, jdata)


@pytest.fixture()
def free_port():
    return get_free_port()


@pytest.mark.unittest
class TestDistGather:
    def test_gather_default(self, free_port):
        world_size = 4
        saver = DistDataSaver()
        mp.spawn(gather_default_worker, args=(world_size, free_port, saver), nprocs=world_size, join=True)

        retval = saver.load_all()
        assert len(retval) == world_size
        tensors = []
        for i in range(len(retval)):
            tensors.append(retval[i]['data'])

        expected_result = torch.concat(tensors)
        for i in range(len(retval)):
            torch.testing.assert_allclose(
                actual=retval[i]['gdata'],
                expected=expected_result,
            )

    def test_gather_default_dim(self, free_port):
        world_size = 4
        saver = DistDataSaver()
        mp.spawn(gather_default_dim_worker, args=(world_size, free_port, saver), nprocs=world_size, join=True)

        retval = saver.load_all()
        assert len(retval) == world_size
        tensors = []
        for i in range(len(retval)):
            tensors.append(retval[i]['data'])

        expected_result = torch.concat(tensors, dim=-1)
        for i in range(len(retval)):
            torch.testing.assert_allclose(
                actual=retval[i]['gdata'],
                expected=expected_result,
            )
