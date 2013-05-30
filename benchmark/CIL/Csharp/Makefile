INSTALLDIR?=/opt/bohrium/
XBUILD?=xbuild
XBUILD_CONF=/property:Configuration=Release

SUBDIRS=nbody shallow_water jacobi blackscholes
 
all:
	for dir in $(SUBDIRS); do \
		$(XBUILD) $(XBUILD_CONF) $$dir/$$dir.sln; \
	done

clean:
	for dir in $(SUBDIRS); do \
		$(XBUILD) $(XBUILD_CONF) $$dir/$$dir.sln /t:clean; \
		rm -rf $$dir/obj $$dir/bin; \
	done
