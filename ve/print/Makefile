BH_BASE=../../
BH_INCLUDE=$(BH_BASE)include/
INCLUDE=-I$(BH_BASE)include
LIB=libbh_ve_print.so
CXX=g++
CC=gcc
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O0 -fPIC $(INCLUDE) $(CFLAGS)
CCFLAGS = $(LCFLAGS) -std=gnu99
CXXFLAGS = $(LCFLAGS) -std=gnu++0x

LDFLAGS = -L../../core -lbh -shared -Wl,-soname,$(LIB)
CPPSRC = bh_ve_print

OBJECTS=$(CPPSRC:=.oo)
INSTALLDIR?=/opt/bh/

all: $(LIB)

%.o: %.c %.h Makefile
		$(CC) -c $< -o $@ $(CCFLAGS)

bh_ve_print.oo: bh_ve_print.cpp Makefile
		$(CXX) -c $< -o $@ $(CXXFLAGS)

%.oo: %.cpp %.hpp Makefile
		$(CXX) -c $< -o $@ $(CXXFLAGS)

$(LIB): $(OBJECTS)
		$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o  *.oo *~ *.gch *.orig $(LIB)

install: all
		cp $(LIB) $(INSTALLDIR)

