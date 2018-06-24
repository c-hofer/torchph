CC=g++

LIB=-L/usr/local/lib\
	-L/usr/local/cuda-9.2/targets/x86_64-linux/lib \
	-L/scratch2/chofer/opt/anaconda3/envs/pyt_gh/lib \
	-L/scratch2/chofer/opt/anaconda3/envs/pyt_gh/lib/python3.6/site-packages/torch/lib \
	-ldl -lprofiler\
	-lcudart \
	-liomp5 -lmklml_intel -lpython3.6m \
	-lcaffe2_gpu -lcaffe2 -lshm -lnccl \
	-L/scratch2/chofer/opt/anaconda3/envs/pyt_gh/lib/python3.6/site-packages/torch\


INC=-I/scratch2/chofer/opt/anaconda3/envs/pyt_gh/lib/python3.6/site-packages/torch/lib/include\
	-I/scratch2/chofer/opt/anaconda3/envs/pyt_gh/include/python3.6m\
	-I/usr/local/cuda-9.2/include
FLAGS=  -Wall -std=c++11 -DPROFILE 


profile: profile_case.o pershom.o pershom_cuda.o 
	$(CC) $(FLAGS) -o profile profile_case.o pershom.o pershom_cuda.o $(INC) $(LIB)

profile_case.o : 
	$(CC) $(FLAGS) -c profile_case.cpp $(INC) $(LIB)

pershom.o: ../../chofer_torchex/pershom/pershom_cpp_src/pershom.cpp
	$(CC) $(FLAGS) -c ../../chofer_torchex/pershom/pershom_cpp_src/pershom.cpp $(INC) $(LIB)

pershom_cuda.o: ../../chofer_torchex/pershom/pershom_cpp_src/pershom_cuda.cu 
	nvcc -std=c++11 -c  ../../chofer_torchex/pershom/pershom_cpp_src/pershom_cuda.cu $(INC) $(LIB)

clean:
	rm -r *.o