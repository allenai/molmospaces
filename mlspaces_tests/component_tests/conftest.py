"""Pytest configuration for component tests.

This module provides shared fixtures and configuration for fast component tests
that don't require expensive simulation or data generation.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_gpu: mark test as requiring GPU (will skip if no GPU available)",
    )


@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    for item in items:
        if "requires_gpu" in item.keywords:
            try:
                import torch

                if not torch.cuda.is_available():
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)
