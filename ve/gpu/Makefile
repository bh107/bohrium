ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_ve_gpu.$(LIB_SUFFIX)
CC?=gcc
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -std=gnu++11 -I$(ROOT)/include -I$(ROOT)/iniparser/src \
         -DBH_TIMING #-DDEBUG 
LDFLAGS= $(ARCH_OPTS) -shared -L../../core -L/usr/lib/nvidia-current -lOpenCL -lbh -pthread -Wl,$(LD_LIB_COMMAND),$(LIB)
SRC=$(ALL_CPP_FILES) $(ALL_C_FILES)
HEADER=$(ALL_H_FILES) $(ALL_HPP_FILES)
OBJC = $(SRC:.c=.o)
OBJ=$(OBJC:.cpp=.o)
INSTALLDIR?=/opt/bohrium/

all: $(LIB)

%.o: %.cpp $(HEADER) Makefile
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

%.o: %.c $(HEADER)  Makefile
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)


$(LIB): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *~ $(LIB)

install: all
	cp $(LIB) $(INSTALLDIR)
	cp -r ocl_source $(INSTALLDIR)/gpu/
	cp -r include $(INSTALLDIR)/gpu/
