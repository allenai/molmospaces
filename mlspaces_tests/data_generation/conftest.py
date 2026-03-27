"""
Pytest configuration for data generation tests.

This module provides session-scoped fixtures for profiling and other shared test setup.
"""

import shutil
import time
from pathlib import Path

import pytest

# Storage for profilers collected during the test session
_profilers = []

# Storage for timing data
_fixture_timings = {}  # {fixture_name: duration}
_test_timings = {}  # {test_nodeid: {'setup': duration, 'call': duration, 'teardown': duration}}
_session_start_time = None


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_debug_images():
    """
    Clean up test_debug_images folder at the start of the test session.

    This ensures we start with a clean slate and don't accumulate debug images
    from previous test runs.
    """
    test_debug_images_dir = Path(__file__).parent / "test_debug_images"
    if test_debug_images_dir.exists():
        shutil.rmtree(test_debug_images_dir)
        print(f"\n[CLEANUP] Cleared test_debug_images folder: {test_debug_images_dir}")
    yield


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="function", autouse=True)
def collect_profilers(request):
    """
    Automatically collect profilers from test configs for session-wide profiling summary.

    This fixture runs for every test function and collects any profiler instances
    from fixtures that have a 'profiler' attribute.
    """
    # After the test runs, check if any fixtures have profilers
    yield

    # Collect profilers from config fixtures
    for fixture_name in ["droid_config", "randomized_config", "rum_config"]:
        if fixture_name in request.fixturenames:
            try:
                config = request.getfixturevalue(fixture_name)
                if hasattr(config, "profiler") and config.profiler is not None:
                    if config.profiler not in _profilers:
                        _profilers.append(config.profiler)
            except Exception:
                # Fixture might not be available for this test
                pass


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    """Track time spent in fixture setup."""
    fixture_name = fixturedef.argname
    start_time = time.perf_counter()

    yield

    duration = time.perf_counter() - start_time

    # Only track fixtures that take meaningful time (>0.1s)
    if duration > 0.1:
        if fixture_name in _fixture_timings:
            _fixture_timings[fixture_name] += duration
        else:
            _fixture_timings[fixture_name] = duration


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Track time spent in test setup."""
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time

    if item.nodeid not in _test_timings:
        _test_timings[item.nodeid] = {}
    _test_timings[item.nodeid]["setup"] = duration


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Track time spent in test execution."""
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time

    if item.nodeid not in _test_timings:
        _test_timings[item.nodeid] = {}
    _test_timings[item.nodeid]["call"] = duration


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    """Track time spent in test teardown."""
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time

    if item.nodeid not in _test_timings:
        _test_timings[item.nodeid] = {}
    _test_timings[item.nodeid]["teardown"] = duration


def pytest_sessionstart(session):
    """Track session start time."""
    global _session_start_time
    _session_start_time = time.perf_counter()


def pytest_sessionfinish(session, exitstatus):
    """
    Called after the entire test session finishes.
    This hook is guaranteed to run and output is always visible.
    """
    from molmo_spaces.utils.test_utils import print_profiling_summary

    # Calculate total session time
    total_session_time = time.perf_counter() - _session_start_time if _session_start_time else 0

    # Build the summary
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("TEST SUITE PROFILING SUMMARY".center(80))
    lines.append("=" * 80)

    # Add overall timing summary
    lines.append(f"\nTotal Session Time: {total_session_time:.2f}s")

    # Add fixture timing summary
    total_fixture_time = 0
    if _fixture_timings:
        lines.append("\n" + "-" * 80)
        lines.append("FIXTURE SETUP TIME".center(80))
        lines.append("-" * 80)

        # Sort fixtures by time (descending)
        sorted_fixtures = sorted(_fixture_timings.items(), key=lambda x: x[1], reverse=True)
        lines.append(f"{'Fixture Name':<50} {'Time':>12}")
        lines.append("-" * 80)

        for fixture_name, duration in sorted_fixtures:
            lines.append(f"{fixture_name:<50} {duration:>11.4f}s")
            total_fixture_time += duration

        lines.append("-" * 80)
        lines.append(f"{'TOTAL FIXTURE TIME':<50} {total_fixture_time:>11.4f}s")

    # Add test timing summary
    if _test_timings:
        lines.append("\n" + "-" * 80)
        lines.append("TEST EXECUTION TIME".center(80))
        lines.append("-" * 80)
        lines.append(f"{'Test':<50} {'Setup':>10} {'Call':>10} {'Teardown':>10}")
        lines.append("-" * 80)

        total_setup = total_call = total_teardown = 0
        for test_id, timings in sorted(_test_timings.items()):
            # Shorten test ID for display
            test_name = test_id.split("::")[-1] if "::" in test_id else test_id
            if len(test_name) > 50:
                test_name = test_name[:47] + "..."

            setup_time = timings.get("setup", 0)
            call_time = timings.get("call", 0)
            teardown_time = timings.get("teardown", 0)

            lines.append(
                f"{test_name:<50} {setup_time:>9.4f}s {call_time:>9.4f}s {teardown_time:>9.4f}s"
            )

            total_setup += setup_time
            total_call += call_time
            total_teardown += teardown_time

        lines.append("-" * 80)
        lines.append(
            f"{'TOTAL TEST TIME':<50} {total_setup:>9.4f}s {total_call:>9.4f}s {total_teardown:>9.4f}s"
        )

        total_test_time = total_setup + total_call + total_teardown
        lines.append(
            f"\nTotal Measured Time (Fixtures + Tests): {total_fixture_time + total_test_time:.2f}s"
        )

        # Calculate overhead/unmeasured time
        overhead = total_session_time - (total_fixture_time + total_test_time)
        if overhead > 0.5:  # Only show if significant
            lines.append(f"Pytest Overhead (collection, hooks, etc.): {overhead:.2f}s")

    # Add detailed profiler summaries
    if _profilers:
        lines.append("\n" + "=" * 80)
        lines.append("DETAILED OPERATION PROFILING".center(80))
        lines.append("=" * 80)
        lines.append(f"Collected {len(_profilers)} profiler(s)")

        # Print summary for each profiler
        profilers_with_data = 0
        for i, profiler in enumerate(_profilers, 1):
            if profiler._avg_time:  # Has data
                profilers_with_data += 1
                if i > 1:
                    lines.append("\n")  # Separator between profilers
                summary = print_profiling_summary(profiler)
                lines.append(summary)

        if profilers_with_data == 0:
            lines.append("\nNote: Profilers were created but no profiling data was collected.")
            lines.append("This may indicate tests didn't run the profiled code paths.")
    else:
        lines.append("\n[No profilers found - profiling may be disabled in test configs]")

    lines.append("\n" + "=" * 80)
    lines.append("END OF PROFILING SUMMARY".center(80))
    lines.append("=" * 80)

    # Output to terminal (always visible, not captured by pytest)
    tw = session.config.get_terminal_writer()
    summary_text = "\n".join(lines)
    tw.line(summary_text, green=True)

    # Also save to file
    output_file = Path("mlspaces_tests/data_generation") / "profiling_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(summary_text)
        f.write(f"\n\nTest exit status: {exitstatus}\n")

    tw.line(f"\nProfiling results also saved to: {output_file.absolute()}", cyan=True)
