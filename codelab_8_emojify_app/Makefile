.PHONY: build
build: venv requirements.txt
	venv/bin/pip-sync
venv:
	python3 -m venv venv
	venv/bin/pip install pip-tools
requirements.txt: requirements.in
	venv/bin/pip-compile requirements.in > requirements.txt