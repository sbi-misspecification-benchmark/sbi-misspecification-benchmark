import csv
from pathlib import Path

import pandas as pd
import pytest

import src.utils.csv_utils as csv_utils


def test_get_csv_header_normal(tmp_path):
    # Create a CSV file with a header and some rows
    p = tmp_path / "test.csv"
    header = ["col1", "col2", "col3"]
    rows = [
        ["a", "b", "c"],
        ["d", "e", "f"]
    ]
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    result = csv_utils.get_csv_header(p)
    assert result == header


def test_get_csv_header_empty(tmp_path):
    # Create an empty file
    p = tmp_path / "empty.csv"
    p.write_text("")  # empty
    with pytest.raises(ValueError) as exc:
        csv_utils.get_csv_header(p)
    # message should mention empty and path
    assert "empty" in str(exc.value)
    assert str(p) in str(exc.value)


def test_assert_csv_header_matches_success(tmp_path):
    # Create a CSV with header
    p = tmp_path / "data.csv"
    expected = ["x", "y"]
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(expected)
    # Should not raise
    csv_utils.assert_csv_header_matches(p, expected)


def test_assert_csv_header_matches_file_not_found(tmp_path):
    p = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError) as exc:
        csv_utils.assert_csv_header_matches(p, ["a"])
    assert "CSV file not found" in str(exc.value)


def test_assert_csv_header_matches_wrong_extension(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("col1,col2\n1,2")
    with pytest.raises(ValueError) as exc:
        csv_utils.assert_csv_header_matches(p, ["col1", "col2"])
    assert "Expected a .csv file" in str(exc.value)


def test_assert_csv_header_matches_mismatch(tmp_path):
    p = tmp_path / "data.csv"
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b"])
    with pytest.raises(ValueError) as exc:
        csv_utils.assert_csv_header_matches(p, ["x", "y"])
    msg = str(exc.value)
    assert "CSV header mismatch" in msg
    assert "got: ['a', 'b']" in msg
    assert "expected: ['x', 'y']" in msg


def test_resolve_file_mode_write(tmp_path):
    p = tmp_path / "any.csv"
    mode, write_header = csv_utils.resolve_file_mode(p, "write")
    assert mode == "w"
    assert write_header is True


def test_resolve_file_mode_append_exists(tmp_path):
    p = tmp_path / "exists.csv"
    p.write_text("dummy")
    mode, write_header = csv_utils.resolve_file_mode(p, "append")
    assert mode == "a"
    assert write_header is False


def test_resolve_file_mode_append_not_exists(tmp_path):
    p = tmp_path / "new.csv"
    if p.exists():
        p.unlink()
    mode, write_header = csv_utils.resolve_file_mode(p, "append")
    assert mode == "w"
    assert write_header is True


def test_resolve_file_mode_invalid():
    with pytest.raises(ValueError) as exc:
        csv_utils.resolve_file_mode(Path("any.csv"), "delete")
    assert "Unknown file_mode" in str(exc.value)


def test_gather_csv_files_direct_file(tmp_path, capsys):
    # Direct existing CSV
    p = tmp_path / "f1.csv"
    p.write_text("header\nvalue")
    result = csv_utils.gather_csv_files(p, base_directory=tmp_path)
    assert isinstance(result, list)
    assert p in result
    # No extra prints
    captured = capsys.readouterr()
    # Should not print "No .csv files found" for this valid file
    assert "No .csv files found" not in captured.out


def test_gather_csv_files_directory(tmp_path, capsys):
    # Create directory with nested CSVs and other files
    d = tmp_path / "dir"
    sub = d / "sub"
    sub.mkdir(parents=True)
    f1 = d / "a.csv"; f1.write_text("h\n1")
    f2 = sub / "b.csv"; f2.write_text("h\n2")
    # also a non-csv
    (sub / "c.txt").write_text("nope")
    result = csv_utils.gather_csv_files(d, base_directory=tmp_path)
    # Should include both CSVs
    assert set(result) == {f1, f2}
    # No error prints for found files
    captured = capsys.readouterr()
    assert "No .csv files found" not in captured.out


def test_gather_csv_files_glob_relative(tmp_path, capsys):
    # Create some CSVs in tmp_path
    f1 = tmp_path / "one.csv"; f1.write_text("h\n1")
    f2 = tmp_path / "two.csv"; f2.write_text("h\n2")
    # Pattern "*.csv"
    pattern = "*.csv"
    result = csv_utils.gather_csv_files(pattern, base_directory=tmp_path)
    # Should find both
    assert set(result) == {f1, f2}
    # No spurious prints
    captured = capsys.readouterr()
    assert "No .csv files found" not in captured.out


def test_gather_csv_files_glob_absolute(tmp_path, capsys):
    # Create CSV in nested directory
    d = tmp_path / "dir"
    d.mkdir()
    f = d / "data.csv"; f.write_text("h\nv")
    pattern = str(d / "*.csv")
    result = csv_utils.gather_csv_files(pattern, base_directory=tmp_path)
    assert set(result) == {f}
    captured = capsys.readouterr()
    assert "No .csv files found" not in captured.out


def test_gather_csv_files_no_match(tmp_path, capsys):
    # Pattern matches nothing
    result = csv_utils.gather_csv_files("no_such_*.csv", base_directory=tmp_path)
    assert result == []
    captured = capsys.readouterr()
    # Should print a message about no matches
    assert "No .csv files found for source" in captured.out


def test_gather_csv_files_type_error():
    with pytest.raises(TypeError):
        csv_utils.gather_csv_files(123, base_directory=Path("."))


def test_read_csv_files_success(tmp_path):
    # Create two CSVs with known content
    f1 = tmp_path / "a.csv"
    df1 = pd.DataFrame({"col": [1, 2]})
    df1.to_csv(f1, index=False)
    f2 = tmp_path / "b.csv"
    df2 = pd.DataFrame({"x": [3, 4]})
    df2.to_csv(f2, index=False)
    paths = [f2, f1]  # unsorted order to test sorting
    frames = csv_utils.read_csv_files(paths)
    # Should return two DataFrames sorted by path name
    sorted_paths = sorted(paths)
    assert len(frames) == 2
    


def test_read_csv_files_skip_error(tmp_path, monkeypatch, capsys):
    # Create one good CSV and one that raises on read
    good = tmp_path / "good.csv"
    pd.DataFrame({"a": [1]}).to_csv(good, index=False)
    bad = tmp_path / "bad.csv"
    bad.write_text("not,a,csv,\x00")  # might still be parseable; instead monkeypatch
    paths = [good, bad]
    # Monkeypatch pd.read_csv to raise for bad.csv
    real_read_csv = pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        if Path(path).name == "bad.csv":
            raise ValueError("simulated read error")
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    frames = csv_utils.read_csv_files(paths)
    # Should only include good.csv
    assert len(frames) == 1
    captured = capsys.readouterr()
    # Should log the failure for bad.csv
    assert "Failed to read" in captured.out
    # Restore pandas.read_csv
    monkeypatch.setattr(pd, "read_csv", real_read_csv)