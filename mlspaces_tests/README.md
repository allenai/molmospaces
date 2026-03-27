# Testing

## Data generation

Run the relevant script, e.g.
```bash
python -m mlspaces_tests.data_generation.generate_test_data_pick
```
which will output the generated data under `mlspaces_tests/test_data/<DATA_TYPE_SUBDIR>`.

## Upload generated data

Assuming you've installed the resource manager, call
```bash
mjt_upload mlspaces_tests/test_data/<DATA_TYPE_SUBDIR> <VERSION_IDENTIFIER_MAYBE_DATE>
```
and ensure the entire contents are to be uploaded as a single archive. If not, please follow the directions output by the script. When the result looks good, rerun with
```bash
mjt_upload mlspaces_tests/test_data/<DATA_TYPE_SUBDIR> <VERSION_IDENTIFIER_MAYBE_DATE> --no-dry-run
```
This last command expects your AWS key pair is set to the proper one for R2 storage, which we are not sharing here.

## Add uploaded data to the resource manager

Edit `molmo_spaces/molmo_spaces_constants.py` and add your new
test data versions. E.g.:
```python
DATA_TYPE_TO_SOURCE_TO_VERSION = dict(
    robots={
        "source_name": "version_string",
        ...
    },

    ...

    test_data={
        "franka_pick": "20251209",
        "franka_pick_and_place": "20251209",
        "rby1_door_opening": "20251209",
        "rum_open_close": "20251209",
        "rum_pick": "20251209",
        "test_randomized_data": "20251209",
        "thormap": "20251209",
    }
)
```

## Running tests

Now you can test as usual, e.g. calling
```bash
rm -r mlspaces_tests/data_generation/test_output
python -m pytest mlspaces_tests/data_generation
```
