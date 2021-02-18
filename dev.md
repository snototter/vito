# Collection of dev commands
* Test and report line coverage:
  ```bash
  pytest --cov-config=tests/.coveragerc --cov=vito/
  ```
* Linting:
  ```bash
  flake8 --max-line-length=127 tests/test_imutils.py
  ```
