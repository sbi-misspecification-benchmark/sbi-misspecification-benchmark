# Python CI-Pipeline with GitHub Actions

This project uses a GitHub Actions CI pipeline to automatically check code quality and run tests on every push or pull request to the `main` branch.

## 🔧 How it works

GItHub Actions runs the CI Pipeline on a seperate server ("runner"-environment), so the workflow must:

1. **Checkout the repository**: The workflow checks out the repository code so it can be tested.
2. **Set up python**: It sets up Python (version 3.11) to ensure consistency.
3. **Install dependencies**: All dependencies listed in `requirements.txt` are installed using `pip`
4. **Run Test and check coverage**: 
   - Tests are run using `pytest` with `coverage` tracking.


## ➕ Adding new Checks

When adding a new CI step (e.g., for security scans or additional tests), follow this basic structure:

```
- name: <Name of the Check>   
  run: |
    <Command 1>
    <Command 2>
    ...
```