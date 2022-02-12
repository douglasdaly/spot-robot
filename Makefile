#
#	Configuration
#
PROJECT_NAME = Squad
PROJECT_SLUG = squad


#
#	Setup
#
RUN_PRE = poetry run

PYTHON = ${RUN_PRE} python
BLACK = ${RUN_PRE} black
COVERAGE = ${RUN_PRE} coverage
FLAKE8 = ${RUN_PRE} flake8
ISORT = ${RUN_PRE} isort
PRE_COMMIT = ${RUN_PRE} pre-commit
PYTEST = ${RUN_PRE} pytest
JUPYTER = ${RUN_PRE} jupyter

PROJECT_DIR = src/${PROJECT_SLUG}/


#
#	Recipes
#
.DEFAULT-GOAL := help-short

# Help
.PHONY: help-short
help-short:
	@printf 'Usage: make \033[36m[target]\033[0m\n'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*? ## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*? ## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ''

.PHONY: help
help:  ## Shows the help menu for all targets.
	@printf 'Usage: make \033[36m[target]\033[0m\n'
	@echo ''
	@echo 'Main targets:'
	@grep -E '^[a-zA-Z_-]+:.*? ## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*? ## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Specific sub-targets:'
	@grep -E '^[a-zA-Z_-]+:.*? ### .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*? ### "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ''

# Environment
.env:  ### Copies the sample environment file to the actual .env file.
	cp sample.env .env

.PHONY: setup
setup: .env setup-venv setup-ipykernel setup-precommit  ## Sets up the development environment.

.PHONY: setup-venv
setup-venv:  ### Sets up the project's development environment.
	poetry install -n

.PHONY: setup-ipykernel
setup-ipykernel:  ### Installs a new IPython kernel for the project's environment.
	${PYTHON} -m ipykernel install --user --name ${PROJECT_SLUG} --display-name '${PROJECT_NAME}'

.PHONY: setup-precommit
setup-precommit:  ### Installs the git pre-commit hooks for this project.
	${PRE_COMMIT} install --install-hooks

.PHONY: update-venv
update-venv:  ### Updates the project's dependencies.
	poetry update -n

.PHONY: update-precommit
update-precommit:  ### Updates the pre-commit hooks to latest versions.
	${PRE_COMMIT} autoupdate

.PHONY: update
update: update-venv update-precommit  ## Updates the project's development environment.

.PHONY: teardown
teardown: teardown-ipykernel teardown-venv  ## Tears down the development environment.

.PHONY: teardown-venv
teardown-venv:  ### Removes the project's development environment.
	rm -rf .venv

.PHONY: teardown-ipykernel
teardown-ipykernel:  ### Uninstalls the project's IPython kernel.
	${JUPYTER} kernelspec remove -f ${PROJECT_SLUG}

# Linting
.PHONY: check-format
check-format:  ### Checks the code's format (w/o changing).
	${BLACK} --check ${PROJECT_DIR} tests/; \
	${ISORT} --check ${PROJECT_DIR} tests/;

.PHONY: check-style
check-style:  ### Checks the code's style.
	${FLAKE8}

.PHONY: format
format:  ## Formats the code in-place.
	${BLACK} ${PROJECT_DIR} tests/; \
	${ISORT} ${PROJECT_DIR} tests/;

.PHONY: lint
lint: check-format check-style  ## Checks the code's format and style.

# Unit testing
.PHONY: test
test:  ## Runs this project's unit tests.
	${PYTEST} tests/ ${PROJECT_DIR}

.PHONY: test-verbose
test-verbose:  ### Runs the project's unit tests with high verbosity.
	${PYTEST} -vv tests/ ${PROJECT_DIR}

.PHONY: test-coverage
test-coverage:  ### Runs the unit tests with coverage reporting on.
	${PYTEST} tests/ ${PROJECT_DIR} --cov=${PROJECT_DIR} --cov-report=xml

.PHONY: coverage
coverage: test-coverage coverage-report  ## Checks the unit tests' code coverage.

.PHONY: coverage-report
coverage-report:  ### Displays the latest code coverage report.
	${COVERAGE} report --skip-covered

# Misc
.PHONY: cloc
cloc:  ## CLOC: Count Lines of Code for the project.
	cloc ${PROJECT_DIR} --exclude-dir=__pycache__
