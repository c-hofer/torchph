CC=g++

LIB=-L/scratch2/chofer/opt/anaconda3/envs/pyt_gh/lib/python3.6/site-packages/torch/lib #-lATen 
INC=-I/scratch2/chofer/opt/anaconda3/envs/pyt_gh/lib/python3.6/site-packages/torch/lib/include \
	-I /scratch2/chofer/opt/anaconda3/envs/pyt_gh/include/python3.6m
FLAGS=  -Wall -std=c++11


lib: pershom.o pershom_cuda.o 
	$(CC) -shared $(FLAGS) -o libpershom.so pershom.o pershom_cuda.o $(INC) $(LIB)

pershom.o: pershom.cpp
	$(CC) -fPIC $(FLAGS) -c pershom.cpp $(INC) $(LIB)

pershom_cuda.o: pershom_cuda.cu 
	nvcc -shared -Xcompiler -fPIC -std=c++11 -c  pershom_cuda.cu $(INC)

clean:
	rm -r *.o

