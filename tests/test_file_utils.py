from pathlib import Path

import src.utils.file_utils as file_utils


def test_ensure_directory_creates_new(tmp_path):
    # Given a non-existing nested directory
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()
    # When calling ensure_directory with Path
    file_utils.ensure_directory(nested)
    # Then the directory should exist
    assert nested.exists() and nested.is_dir()


def test_ensure_directory_with_str(tmp_path):
    # Given a non-existing directory as string
    nested_str = str(tmp_path / "x" / "y")
    # When calling ensure_directory with string
    file_utils.ensure_directory(nested_str)
    # Then it should create the directory
    created = Path(nested_str)
    assert created.exists() and created.is_dir()


def test_ensure_directory_already_exists(tmp_path):
    # Given an existing directory
    d = tmp_path / "existing"
    d.mkdir()
    # Should not raise
    file_utils.ensure_directory(d)
    # Still exists
    assert d.exists() and d.is_dir()


def test_unique_path_no_collision(tmp_path):
    # Given a path that does not exist
    p = tmp_path / "file.txt"
    # unique_path should return the same path
    result = file_utils.unique_path(p)
    assert result == p
    assert not result.exists()


def test_unique_path_single_collision(tmp_path):
    # Create an existing file
    p = tmp_path / "data.csv"
    p.write_text("content")
    # unique_path should return with suffix __1 before extension
    result = file_utils.unique_path(p, sep="__")
    expected = tmp_path / "data__1.csv"
    assert result == expected
    assert not expected.exists()


def test_unique_path_multiple_collisions(tmp_path):
    # Create files data.txt, data__1.txt, data__2.txt
    base = tmp_path / "data.txt"
    base.write_text("x")
    f1 = tmp_path / "data__1.txt"
    f1.write_text("x")
    f2 = tmp_path / "data__2.txt"
    f2.write_text("x")
    # unique_path should return data__3.txt
    result = file_utils.unique_path(base, sep="__")
    expected = tmp_path / "data__3.txt"
    assert result == expected
    assert not expected.exists()


def test_unique_path_custom_sep(tmp_path):
    # Test with custom separator
    p = tmp_path / "report.log"
    p.write_text("log")
    result = file_utils.unique_path(p, sep="-")
    expected = tmp_path / "report-1.log"
    assert result == expected


def test_unique_path_directory_collision(tmp_path):
    # If directory name matches pattern: unique_path is for files, so if path is directory and exists, 
    # candidate will be directory-like but unique_path only checks existence: test behavior
    d = tmp_path / "dir"
    d.mkdir()
    # unique_path on directory returns path__1
    result = file_utils.unique_path(d, sep="__")
    expected = tmp_path / "dir__1"
    assert result == expected
    assert not expected.exists()


def test_create_unique_path_creates_directory(tmp_path):
    # Given a nested directory that doesn't exist and no existing file
    dir_path = tmp_path / "newdir"
    stem = "file"
    # Call create_unique_path
    result = file_utils.create_unique_path(dir_path, stem, extension=".txt", sep="__")
    # Directory should now exist
    assert dir_path.exists() and dir_path.is_dir()
    # The returned path should be dir_path/file.txt
    expected = dir_path / "file.txt"
    assert result == expected
    assert not expected.exists()


def test_create_unique_path_collision(tmp_path):
    # Given directory exists and a file with the target name exists
    dir_path = tmp_path / "d"
    dir_path.mkdir()
    existing = dir_path / "report.csv"
    existing.write_text("data")
    # Also create report__1.csv to simulate multiple collisions
    coll1 = dir_path / "report__1.csv"
    coll1.write_text("more")
    # Call create_unique_path: should return report__2.csv
    result = file_utils.create_unique_path(dir_path, "report", extension=".csv", sep="__")
    expected = dir_path / "report__2.csv"
    assert result == expected
    assert not expected.exists()


def test_create_unique_path_with_str_directory(tmp_path):
    # Directory as string
    dir_str = str(tmp_path / "strdir")
    stem = "test"
    result = file_utils.create_unique_path(dir_str, stem, extension=".dat", sep="__")
    # Directory should be created
    created = Path(dir_str)
    assert created.exists() and created.is_dir()
    expected = created / "test.dat"
    assert result == expected
    assert not expected.exists()


def test_unique_path_on_deeply_nested(tmp_path):
    # Create nested dirs and files with same stem
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    base = nested / "item.txt"
    base.write_text("1")
    # unique_path should respect nested directory
    result = file_utils.unique_path(base)
    expected = nested / "item_1.txt"  # default sep is "_" per signature, but doc says "__"? 
    # Actually default sep in signature is "_" so use "_"
    assert result == expected


def test_unique_path_non_strange_extensions(tmp_path):
    # Test file with no extension
    base = tmp_path / "filename"
    base.write_text("x")
    result = file_utils.unique_path(base, sep="-")
    expected = tmp_path / "filename-1"
    assert result == expected
