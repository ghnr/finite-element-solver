# Finite element solver

Performs a finite element analysis of a structure in 3D space when given the input nodes, elements (12 total DOF), forces, and boundary conditions using the Eigen library.
Solves for reaction forces and nodal displacements from the computed sparse matrix. Also outputs data to a `vtk` file for visualisation of the result structure in Paraview.

Example inputs are provided which represent the structure below:

![example structure](https://github.com/ghnr/finite-element-solver/blob/master/docs/structure.png)
