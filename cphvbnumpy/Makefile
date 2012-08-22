CPHVB_PYTHON?=python
INSTALLDIR?=/opt/cphvb/

all:
	$(CPHVB_PYTHON) setup.py build

clean:
	rm -Rf build

install:
	$(CPHVB_PYTHON) setup.py install --prefix $(INSTALLDIR)
