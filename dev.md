# Collection of dev commands
* Test and report line coverage:
  ```bash
  pip install pytest-cov
  pytest --cov-config=tests/.coveragerc --cov=vito/
  pytest --cov-config=tests/.coveragerc --cov=vito/ --cov-report term-missing
  ```
* Linting:
  ```bash
  flake8 --max-line-length=127 tests/test_imutils.py
  ```
