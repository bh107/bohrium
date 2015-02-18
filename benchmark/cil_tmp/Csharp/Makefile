ROOT=../../..
include $(ROOT)/platform.mk
XBUILD_CONF=/property:Configuration=Release /verbosity:minimal /nologo


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
