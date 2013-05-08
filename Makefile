BH_PYTHON?=python

all:
	./build.py build --interpreter=$(BH_PYTHON)

clean:
	./build.py clean

install:
ifdef DESTDIR
	./build.py install --prefix=$(DESTDIR) --interpreter=$(BH_PYTHON)
else
	./build.py install --interactive --interpreter=$(BH_PYTHON)
endif