# MATRIX MULTIPLICATION SCALING STUDY

The following is a comparison of the performance of distributed matrix multiplication. Three different implementations of matrix multiplication are compared: 
- Naive
- Cblas
- Cublas

Matrices are always initialized on CPU on purpose.

# Compile 
To compile the program
> bash jobs/compile.sh


# RUN

To run the scaling test
> bash jobs/scal.sh `Matrix_size` `cpu|gpu`

if `cpu` than set additional argument `0|1` to specify **Naive** or **Cblas** implementation. 

