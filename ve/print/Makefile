CPHVB_BASE=../../
CPHVB_INCLUDE=$(CPHVB_BASE)include/
INCLUDE=-I$(CPHVB_BASE)include
LIB=libcphvb_ve_print.so
CXX=g++
CC=gcc
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O0 -fPIC $(INCLUDE) $(CFLAGS)
CCFLAGS = $(LCFLAGS) -std=gnu99
CXXFLAGS = $(LCFLAGS) -std=gnu++0x

LDFLAGS = -L../../core -lcphvb -shared -Wl,-soname,$(LIB)
CPPSRC = cphvb_ve_print

OBJECTS=$(CPPSRC:=.oo)
INSTALLDIR?=/opt/cphvb/

all: $(LIB)

%.o: %.c %.h Makefile
		$(CC) -c $< -o $@ $(CCFLAGS)

cphvb_ve_print.oo: cphvb_ve_print.cpp Makefile
		$(CXX) -c $< -o $@ $(CXXFLAGS)

%.oo: %.cpp %.hpp Makefile
		$(CXX) -c $< -o $@ $(CXXFLAGS)

$(LIB): $(OBJECTS)
		$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o  *.oo *~ *.gch *.orig $(LIB)

install: all
		cp $(LIB) $(INSTALLDIR)

