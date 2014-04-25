docs:
	mkdir -p doc/output
	doxygen doc/cpu_engine.doxy

gen:
	cd autogen && ./gen.py
	cp autogen/output/block_compose.cpp .
	cp autogen/output/utils_mapping.cpp .
	cp autogen/output/specializer_cexpression.cpp .
	cp autogen/output/tac.cpp tac.h
	rm autogen/output/*.cpp

purge:
	rm -rf ~/.local/share/bh/templates
	rm -rf ~/.local/var/bh/kernels
	rm -rf ~/.local/var/bh/objects
	rm -rf ~/.local/var/bh/scripts
	rm ~/.local/lib/libbh_ve_cpu.so
	rm ~/.local/include/tac.h

reset: purge
	mkdir bdir
	cd bdir && cmake ../../../ -DCMAKE_INSTALL_PREFIX=~/.local && make && make install

move_fuse:
	python tools/move_code.py ~/.local/var/bh/kernels/ /tmp/code/fuse

move_sij:
	python tools/move_code.py ~/.local/var/bh/kernels/ /tmp/code/sij

prep_fuse:

