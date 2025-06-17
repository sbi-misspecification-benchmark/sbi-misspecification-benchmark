import pytest
import csv
import src.utils.csv_utils as csv_utils


def test_get_csv_header_and_assert_header_matches(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError):
        csv_utils.get_csv_header(empty)

    f = tmp_path / "h.csv"
    with f.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["a", "b"])
        writer.writerow(["1", "2"])
    assert csv_utils.get_csv_header(f) == ["a", "b"]

    # matching
    csv_utils.assert_csv_header_matches(f, ["a", "b"])

    # mismatch
    with pytest.raises(ValueError):
        csv_utils.assert_csv_header_matches(f, ["x", "y"])


def test_merge_csv_rows(tmp_path):
    folder = tmp_path / "csvs"
    folder.mkdir()

    # create two CSVs
    f1 = folder / "a.csv"
    with f1.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["x", "y"])
        writer.writerow(["1", "2"])

    f2 = folder / "b.csv"
    with f2.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["x", "y"])
        writer.writerow(["3", "4"])
    rows = csv_utils.merge_csv_rows(folder)
    assert rows == [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}]

    # header validation
    with pytest.raises(ValueError):
        csv_utils.merge_csv_rows(folder, expected_header=["wrong"])
