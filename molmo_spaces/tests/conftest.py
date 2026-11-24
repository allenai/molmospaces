import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for datagen tests",
    )
    parser.addoption(
        "--push-wandb",
        action="store_true",
        help="Whether or not to push the results to wandb",
    )


@pytest.fixture
def num_workers(request):
    return request.config.getoption("--num-workers")


@pytest.fixture
def push_wandb(request):
    return request.config.getoption("--push-wandb")
