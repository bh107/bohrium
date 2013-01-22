BH_PYTHON?=python
INSTALLDIR?=/opt/bh/

all:
	$(BH_PYTHON) setup.py build

clean:
	rm -Rf build

install:
	$(BH_PYTHON) setup.py install --prefix $(INSTALLDIR)
