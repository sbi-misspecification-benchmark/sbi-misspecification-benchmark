import pytest
import src.utils.file_utils as file_utils


def test_ensure_directory(tmp_path):
    new_dir = tmp_path / "nested" / "dir"
    assert not new_dir.exists()

    file_utils.ensure_directory(new_dir)
    assert new_dir.exists() and new_dir.is_dir()


def test_unique_path(tmp_path):
    p = tmp_path / "foo.txt"
    # no existing file
    assert file_utils.unique_path(p) == p

    # first collision
    p.write_text("x")
    p1 = tmp_path / "foo__1.txt"
    assert file_utils.unique_path(p) == p1

    # second collision
    p1.write_text("y")
    p2 = tmp_path / "foo__2.txt"
    assert file_utils.unique_path(p) == p2


def test_resolve_write_mode(tmp_path):
    p = tmp_path / "f.csv"
    # write mode always creates new
    assert file_utils.resolve_write_mode(p, "write") == ("w", True)

    # append mode, missing file => write
    if p.exists():
        p.unlink()
    assert file_utils.resolve_write_mode(p, "append") == ("w", True)

    # append mode, existing file => append without the header
    p.write_text("hi")
    assert file_utils.resolve_write_mode(p, "append") == ("a", False)

    # invalid write_mode
    with pytest.raises(ValueError):
        file_utils.resolve_write_mode(p, "invalid")
