all: cpp cuda
cuda:
	 nvcc glob.cu -o glob; nvcc shmem.cu -o shmem
cpp:
	cp glob.cpp ./glob.cu; cp shmem.cpp ./shmem.cu
