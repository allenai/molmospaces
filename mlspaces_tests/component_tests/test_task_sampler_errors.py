from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask


def test_positional_message_is_the_reason():
    e = HouseInvalidForTask("No pickup candidates found in the scene")

    assert e.reason == "No pickup candidates found in the scene"
    assert str(e) == "House invalid for tasks: No pickup candidates found in the scene"


def test_reason_and_details():
    e = HouseInvalidForTask(
        reason="MuJoCo compilation failed due to unstable physics",
        house_info="Compilation error: Nan, Inf or huge value",
    )

    assert str(e) == (
        "House invalid for tasks: MuJoCo compilation failed due to unstable physics. "
        "Details: Compilation error: Nan, Inf or huge value"
    )


def test_details_alone_are_still_reported():
    e = HouseInvalidForTask(house_info="Setup error: bad mesh")

    assert str(e) == "House invalid for tasks: no reason given. Details: Setup error: bad mesh"


def test_an_error_alone_becomes_the_reason():
    e = HouseInvalidForTask(error=ValueError("bad inertia"))

    assert e.reason == "ValueError: bad inertia"
    assert "ValueError: bad inertia" in str(e)


def test_no_arguments():
    assert str(HouseInvalidForTask()) == "House invalid for tasks: no reason given"
