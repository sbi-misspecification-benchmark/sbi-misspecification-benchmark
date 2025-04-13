# Preparing a release

## Decide what will be the upcoming version number

- `sbi benchmark` currently uses the [Semver 2.0.0](https://semver.org/) convention.
- Edit the version number in the tuple at `sbi/sbi/__version__.py`.

## Collect a list of relevant changes

- [ ] Edit `changelog.md`: Add a new version number header and report changes below it.

  Trick: To get a list of all changes since the last PR, you can start creating a
  release via GitHub already, add a
  tag and then let GitHub automatically draft the release notes. Note that some changes
  might not be worth mentioning, or others might be missing or needing more explanation.
- [ ] Use one line per change, include links to the pull requests that implemented each of
  the changes.
- [ ] **Credit contributors**!
- [ ] If there are new package dependencies or updated version constraints for the existing
  dependencies, add/modify the corresponding entries in `pyproject.toml`.
- [ ] Test the installation in a fresh conda env to make sure all dependencies match.

## Run tests locally and make sure they pass

- Run the **full test suite, including slow tests.**
  - [ ] slow tests are passing
  - [ ] GPU tests are passing


