CC  = gcc
CXX = g++
CCFLAGS = $(LCFLAGS) -std=gnu99
CXXFLAGS = $(LCFLAGS) -std=gnu++0x
LIBS = -lcphvb -lcphvb_vem_node
LIBS += -lcphvb_ve_simple 
#LIBS += -lcphvb_ve_cuda -lcuda 
INCLUDE=-I../include

all: simple_frontend

simple_frontend: simple_frontend.c
		$(CC) $(CCFLAGS) $(INCLUDE) $(LIBS) -o $@ $<

clean:
		rm -f simple_frontend *~
