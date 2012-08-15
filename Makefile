CPHVB_PYTHON?=python

all:
	./build.py build --interpreter=$(CPHVB_PYTHON)

clean:
	./build.py clean

install:
	./build.py install --interactive --interpreter=$(CPHVB_PYTHON)
