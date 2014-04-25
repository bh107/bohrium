CC = mpicc
#CC = gcc
CPP = g++
FLAGS = -O2 -lm
BOOST = /opt/local/include/ 

all: heat-eq-jacobi jacobi-naive jacobi-simple jacobi-tuned jacobi-vtuned jacobi-boost

bench: all
	./jacobi.sh

clean:
	rm -f jacobi-naive jacobi-simple jacobi-tuned jacobi-vtuned jacobi-boost *~

heat-eq-jacobi: src/heat-eq-jacobi.c
	$(CC) src/heat-eq-jacobi.c -o bin/heat-eq-jacobi -O2 -lm -fopenmp -std=gnu99

jacobi-naive: src/jacobi-naive.c
	$(CC) src/jacobi-naive.c -o bin/jacobi-naive $(FLAGS)

jacobi-simple: src/jacobi-simple.c
	$(CC) src/jacobi-simple.c -o bin/jacobi-simple $(FLAGS)

jacobi-tuned: src/jacobi-tuned.c
	$(CC) src/jacobi-tuned.c -o bin/jacobi-tuned $(FLAGS)

jacobi-vtuned: src/jacobi-vtuned.c
	$(CC) src/jacobi-vtuned.c -o bin/jacobi-vtuned $(FLAGS)

jacobi-boost: src/jacobi-boost.cc
	$(CPP) -I$(BOOST) src/jacobi-boost.cc -o bin/jacobi-boost $(FLAGS)
