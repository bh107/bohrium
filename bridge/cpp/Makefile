INSTALLDIR?=/opt/bohrium/

gen:
	./codegen/gen.py
	mv codegen/output/*.hpp bh/.
	@-(echo 'Bytecode based autogeneration is done.')

install: gen
	mkdir -p $(INSTALLDIR)/include
	cp -r bh $(INSTALLDIR)/include/
	@-(echo 'Installation is done.')

clean:
	rm -rf *.o *~ codegen/output/*.hpp
	rm -rf bin/
	@-(echo 'Cleaning is done.')

