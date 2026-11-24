import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for datagen tests",
    )


@pytest.fixture
def num_workers(request):
    return request.config.getoption("--num-workers")
