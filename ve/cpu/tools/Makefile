ROOT=../../..
CPU=$(ROOT)/ve/cpu
BDIR=$(CPU)/bdir

docs:
	mkdir -p $(CPU)/doc/output
	cd $(CPU) && doxygen doc/cpu_engine.doxy

gen:
	cd $(CPU)/autogen && ./gen.py
	cp $(CPU)/autogen/output/block_compose.cpp $(CPU)/
	cp $(CPU)/autogen/output/utils_mapping.cpp $(CPU)/
	cp $(CPU)/autogen/output/specializer_cexpression.cpp $(CPU)/
	cp $(CPU)/autogen/output/tac.cpp $(CPU)/tac.h
	rm $(CPU)/autogen/output/*.cpp

purge:
	rm -rf ~/.local/share/bh/templates
	rm -rf ~/.local/var/bh/kernels
	rm -rf ~/.local/var/bh/objects
	rm -rf ~/.local/var/bh/scripts
	rm -f ~/.local/lib/libbh_ve_cpu.so
	rm -f ~/.local/include/tac.h

reset: purge install

install: gen
	mkdir -p $(BDIR)
	cd $(BDIR) && cmake $(ROOT) -DCMAKE_INSTALL_PREFIX=~/.local && make && make install

move_fuse:
	python $(CPU)/tools/move_code.py ~/.local/var/bh/kernels/ /tmp/code/fuse

move_sij:
	python $(CPU)/tools/move_code.py ~/.local/var/bh/kernels/ /tmp/code/sij

prep_fuse:
	mkdir -p /tmp/code/fuse
	rm /tmp/code/fuse/*.c

prep_sij:
	mkdir -p /tmp/code/sij
	rm /tmp/code/sij/*.c

clean:
	rm -rf $(BDIR)
