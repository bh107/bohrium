CC = gcc
CCFLAGS = $(LCFLAGS) -std=gnu99
LIBS = -lcphvb
LIBS += -lcphvb_vem_node
LIBS += -lcphvb_ve_simple 
LIBS += -lcphvb_ve_score 
LIBS += -lcphvb_ve_mcore 
LIBS += -lpthread
LIBS += -lrt
LIBS += -lc
#LIBS += -lcphvb_ve_cuda 
SRC=$(subst .c,,$(wildcard *.c))
INCLUDE=-I../include
HANSI=-L/usr/local/lib

all: $(SRC)

%: %.c
	$(CC) $(CCFLAGS) $(HANSI) $(INCLUDE) -o $@.test $< $(LIBS) 

clean:
	rm -f test_*.test *~
