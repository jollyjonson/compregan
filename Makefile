.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")
USING_POETRY=$(shell grep "tool.poetry" pyproject.toml && echo "yes")

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: install
install:          ## Install the project in dev mode.
	@if [ "$(USING_POETRY)" ]; then poetry install && exit; fi
	@echo "Don't forget to run 'make virtualenv' if you got errors."
	$(ENV_PREFIX)pip install -e .[test]

.PHONY: fmt
fmt:              ## Format code using black & isort.
	$(ENV_PREFIX)isort compregan/
	$(ENV_PREFIX)black -l 79 compregan/
	$(ENV_PREFIX)black -l 79 test/
	$(ENV_PREFIX)black -l 79 experiments/

.PHONY: lint
lint:             ## Run pep8, black, mypy linters.
	$(ENV_PREFIX)flake8 compregan/
	$(ENV_PREFIX)black -l 79 --check compregan/
	$(ENV_PREFIX)black -l 79 --check test/
	$(ENV_PREFIX)black -l 79 --check experiments/
	$(ENV_PREFIX)mypy --ignore-missing-imports compregan/

.PHONY: test
test: lint        ## Run tests and generate coverage report.
	$(ENV_PREFIX)pytest -v --cov-config test/.coveragerc --cov=compregan -l --tb=short --maxfail=1 test/
	$(ENV_PREFIX)coverage xml
	$(ENV_PREFIX)coverage html

.PHONY: watch
watch:            ## Run tests on every change.
	ls **/**.py | entr $(ENV_PREFIX)pytest -s -vvv -l --tb=long --maxfail=1 test/

.PHONY: clean
clean:            ## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build

.PHONY: release
release:          ## Create a new tag for release.
	@echo "WARNING: This operation will create s version tag and push to github"
	@read -p "Version? (provide the next x.y.z semver) : " TAG
	@echo "$${TAG}" > compregan/VERSION
	@$(ENV_PREFIX)gitchangelog > HISTORY.md
	@git add compregan/VERSION HISTORY.md
	@git commit -m "release: version $${TAG} 🚀"
	@echo "creating git tag : $${TAG}"
	@git tag $${TAG}
	@git push -u origin HEAD --tags
	@echo "Github Actions will detect the new tag and release the new version."

#.PHONY: docs
#docs:             ## Build the documentation.
#	@echo "building documentation ..."
#	@$(ENV_PREFIX)mkdocs build
#	URL="site/index.html"; xdg-open $$URL || sensible-browser $$URL || x-www-browser $$URL || gnome-open $$URL

#.PHONY: switch-to-poetry
#switch-to-poetry: ## Switch to poetry package manager.
#	@echo "Switching to poetry ..."
#	@if ! poetry --version > /dev/null; then echo 'poetry is required, install from https://python-poetry.org/'; exit 1; fi
#	@rm -rf .venv
#	@poetry init --no-interaction --name=a_flask_test --author=rochacbruno
#	@echo "" >> pyproject.toml
#	@echo "[tool.poetry.scripts]" >> pyproject.toml
#	@echo "compregan = 'compregan.__main__:main'" >> pyproject.toml
#	@cat requirements.txt | while read in; do poetry add --no-interaction "$${in}"; done
#	@cat requirements-test.txt | while read in; do poetry add --no-interaction "$${in}" --dev; done
#	@poetry install --no-interaction
#	@mkdir -p .github/backup
#	@mv requirements* .github/backup
#	@mv setup.py .github/backup
#	@echo "You have switched to https://python-poetry.org/ package manager."
#  	@echo "Please run 'poetry shell' or 'poetry run compregan'"
