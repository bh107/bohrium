CPHVB_PYTHON?=python

all:
	$(CPHVB_PYTHON) build.py build

clean:
	$(CPHVB_PYTHON) build.py clean

install:
	$(CPHVB_PYTHON) build.py install --interactive
