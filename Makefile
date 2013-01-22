BH_PYTHON?=python

all:
	./build.py build --interpreter=$(BH_PYTHON)

clean:
	./build.py clean

install:
	./build.py install --interactive --interpreter=$(BH_PYTHON)
