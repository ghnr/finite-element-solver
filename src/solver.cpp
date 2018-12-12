#include <iostream>
#include <math.h>
#include <array>
#include <algorithm>
#include <Eigen>
#include <unordered_set>
#include <fstream>
#include <iterator>
#include <chrono>

using namespace Eigen;
using std::vector;


struct node_struct {
    double x,y,z;
    node_struct(double coor_x, double coor_y, double coor_z) {
        x = coor_x;
        y = coor_y;
        z = coor_z;
    }
};

void parseInput(vector<node_struct>& nodes, vector<vector<int>>& elements, std::unordered_set<int>& fixed_indices_hash,
                VectorXd& disp_full, int& gDOF);

void stiffnessMatrixBeam(const vector<node_struct>& nodes, const vector<vector<int>>& elements, const int gDOF,
                         const std::unordered_set<int>& fixed_DOF, vector<Triplet<double>>& sparse_ijv,
                         vector<Triplet<double>>& sparse_ijv_afterBC, vector<Triplet<double>>& sparse_ijv_fixed);

VectorXd solve(const std::unordered_set<int>& fixed_indices_hash, const vector<Triplet<double>>& sparse_vector,
           const vector<Triplet<double>>& sparse_vector_free, const vector<Triplet<double>>& sparse_vector_fixed, VectorXd& disp_full,
           const int gDOF);

void save_to_vtk(const vector<node_struct>& nodes, const vector<vector<int>>& elements, VectorXd& disp_full,
                 VectorXd& forces_full, const int gDOF);

int main() {
    // Use the STL clock method to measure execution time of the program
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    // Define variables to be used throughout the code
    vector<node_struct> nodes;
    vector<vector<int>> elements;
    // Hashset used for O(1) lookup that will store all of the global matrix indices associated with the fixed DOF in certain nodes
    std::unordered_set<int> fixed_indices_hash;
    // Vectors that use the Triplet template that hold row, column and value vector from which the sparse matrix is made
    // sparse_vector holds the values that will make the full unrestricted stiffness matrix
    vector<Triplet<double>> sparse_vector;
    // Corresponds to values after the boundary conditions have been applied, used to calculate displacements
    vector<Triplet<double>> sparse_vector_free;
    // Corresponds to values for only the fixed DOFs, used to calculate reaction forces at those points
    vector<Triplet<double>> sparse_vector_fixed;
    VectorXd disp_full;
    // 6 dof per node: 3 in translation and 3 in rotation, so this will be 6 * number of nodes
    int gDOF;

    // Assign values to nodes and elements vectors from reading input .txt files. Most parameters are passed by reference to the functions below
    // mostly out of necessity but also to avoid unnecessary copies
    parseInput(nodes, elements, fixed_indices_hash, disp_full, gDOF);

    // Calculate all 3 of the Eigen::Triplets by iterating over all of the elements and applying the relevant beam equations in 3D
    stiffnessMatrixBeam(nodes,elements,gDOF,fixed_indices_hash,sparse_vector,sparse_vector_free,sparse_vector_fixed);

    // Compute and solve displacement and reaction forces from the sparse matrices defined above and return them as a pair
    VectorXd forces_full = solve(fixed_indices_hash, sparse_vector, sparse_vector_free, sparse_vector_fixed, disp_full, gDOF);

    // The results returned from the solve function are now saved to a .vtk file for post-processing
    save_to_vtk(nodes,elements,disp_full,forces_full,gDOF);

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::cout << "Execution time (sec) = " << (std::chrono::duration_cast<std::chrono::microseconds>
            (end_time - start_time).count()) /1000000.0 <<std::endl;
}


void parseInput(vector<node_struct>& nodes, vector<vector<int>>& elements, std::unordered_set<int>& fixed_indices_hash,
                VectorXd& disp_full, int& gDOF) {

    std::cout << "Reading files..." << std::endl;
    std::ifstream node_file("nodes.txt");
    std::ifstream element_file("elements.txt");
    std::ifstream bc_file("boundary_conditions.txt");

    double x, y, z;
    while (node_file >> x >> y >> z) {
        nodes.push_back(node_struct(x,y,z));
    }
    node_file.close();

    int e1, e2;
    while (element_file >> e1 >> e2) {
        elements.push_back({e1,e2});
    }
    element_file.close();

    gDOF = nodes.size() * 6;
    disp_full = VectorXd::Ones(gDOF);

    std::string line;
    while (std::getline (bc_file, line)) {
        int num;
        std::stringstream iss(line);
        vector<int> line_integers;
        while (iss >> num) {
            line_integers.push_back(num);
        }
        int node = line_integers[0];
        for (int i=1; i<7; i++) {
            if (line_integers[i] == 0) {
                int global_index = node*6 + (i-1);
                fixed_indices_hash.insert(global_index);
                disp_full(global_index) = 0;
            }
        }
    }
    bc_file.close();
    std::cout << "Structure with " << nodes.size() << " nodes and " << elements.size() << " elements found" << std::endl;
}



void stiffnessMatrixBeam(const vector<node_struct>& nodes, const vector<vector<int>>& elements, int gDOF,
                         const std::unordered_set<int>& fixed_DOF, vector<Triplet<double>>& sparse_ijv,
                         vector<Triplet<double>>& sparse_ijv_afterBC, vector<Triplet<double>>& sparse_ijv_fixed) {

    std::cout << "Creating global matrix..." << std::endl;
    SparseMatrix<double> global_matrix(gDOF,gDOF);

    vector<int> adjust_indices(gDOF,0);

    for (int bc : fixed_DOF) {
        for (int i=bc; i<adjust_indices.size(); i++) {
            adjust_indices[i] += 1;
        }
    }

    // 5mmx5mm square cross-section
    const double pi = std::atan(1.0)*4;
    const int SQUARE = 0;
    const int CIRCLE = 1;
    const int HEXAGON = 2;
    const int EQUILATERAL = 3;

    double E = 65e9;
    double G = 40e9;
    double v = (E/G)/2 - 1;

    float side;
    double A;
    double J;
    double Iz;
    double Iy;
    double h;
    
    // replace with exacts
    switch (SQUARE) {
        case SQUARE:
            side = 0.005;
            A = pow(side,2);
            J = 2.25 * pow(side,4);
            Iz = pow(side,4)/12;
            Iy = pow(side,4)/12;
            break;
        case CIRCLE:
            side = 0.00564;
            A = (pi*pow(side,2))/4;
            J = (pi*pow(side,4))/32;
            Iz = (pi*pow(side/2,4))/4;
            Iy = (pi*pow(side/2,4))/4;
            break;
        case HEXAGON:
            side = 0.0031;
            A = (pow(side,2)*3*sqrt(3))/2;
            h = sqrt(3)*side;
            J = 0.1154*pow(h,4);
            Iz = (pow(side,4)*5*sqrt(3))/16;
            Iy = (pow(side,4)*5*sqrt(3))/16;
            break;
        case EQUILATERAL:
            side = 0.007598;
            A = (sqrt(3)*pow(side,2))/4;
            J = (pow(side,4)*sqrt(3))/80;
            Iz = 0.01804*pow(side,4);
            Iy = 0.01804*pow(side,4);
            break;
        default:
            A = 1;
            J = 1;
            Iz = 1;
            Iy = 1;
            break;
    }

    for (int e=0; e<elements.size(); e++) {

        int el_n1 = elements[e][0];
        int el_n2 = elements[e][1];

        double x1 = nodes[el_n1].x;
        double y1 = nodes[el_n1].y;
        double z1 = nodes[el_n1].z;
        double x2 = nodes[el_n2].x;
        double y2 = nodes[el_n2].y;
        double z2 = nodes[el_n2].z;

        double L = sqrt(pow(x2-x1, 2) + pow(y2-y1, 2) + pow(z2-z1, 2));

        double l = (x2-x1)/L;
        double m = (y2-y1)/L;
        double n = (z2-z1)/L;
        double D = sqrt(pow(l,2) + pow(m,2));

        MatrixXf lambda(3,3);

        if (x2 == x1 && y2 == y1) {
            if (z2 > z1) {
                lambda << 0,0,1,
                        0,1,0,
                        -1,0,0;
            } else {
                lambda << 0,0,-1,
                        0,1,0,
                        1,0,0;
            }
        } else {
            lambda << l,m,n,
                    -m/D,l/D,0,
                    -(l*n)/D,-(m*n)/D,D;
        }

        MatrixXf T = MatrixXf::Zero(12,12);

        for (int x=0; x<4; x++) {
             T.block<3,3>(3*x,3*x) = lambda;
        }

        MatrixXf k(12,12);

        // This is the standard 12x12 matrix for a beam element in 3D with Young's Modulus factored out for brevity's sake

        k << A/L, 0, 0, 0, 0, 0, -A/L, 0, 0, 0, 0, 0,
                0, 12*Iz/(L*L*L), 0, 0, 0, 6*Iz/(L*L), 0, -12*Iz/(L*L*L), 0, 0, 0, 6*Iz/(L*L),
                0, 0, 12*Iy/(L*L*L), 0, -6*Iy/(L*L), 0, 0, 0, -12*Iy/(L*L*L), 0, -6*Iy/(L*L), 0,
                0, 0, 0, J/(2*(1+v)*L), 0, 0, 0, 0, 0, -J/(2*(1+v)*L), 0, 0,
                0, 0, -6*Iy/(L*L), 0, 4*Iy/L, 0, 0, 0, 6*Iy/(L*L), 0, 2*Iy/L, 0,
                0, 6*Iz/(L*L), 0, 0, 0, 4*Iz/L, 0, -6*Iz/(L*L), 0, 0, 0, 2*Iz/L,
                -A/L, 0, 0, 0, 0, 0, A/L, 0, 0, 0, 0, 0,
                0, -12*Iz/(L*L*L), 0, 0, 0, -6*Iz/(L*L), 0, 12*Iz/(L*L*L), 0, 0, 0, -6*Iz/(L*L),
                0, 0, -12*Iy/(L*L*L), 0, 6*Iy/(L*L), 0, 0, 0, 12*Iy/(L*L*L), 0, 6*Iy/(L*L), 0,
                0, 0, 0, -J/(2*(1+v)*L), 0, 0, 0, 0, 0, J/(2*(1+v)*L), 0, 0,
                0, 0, -6*Iy/(L*L), 0, 2*Iy/L, 0, 0, 0, 6*Iy/(L*L), 0, 4*Iy/L, 0,
                0, 6*Iz/(L*L), 0, 0, 0, 2*Iz/L, 0, -6*Iz/(L*L), 0, 0, 0, 4*Iz/L;

        k *= E;

        k = T.transpose() * k * T;

        // This uses a formula to translate the local stiffness matrix to the global one. It does this by looping
        // through every column and row and if there is a non-zero value, then place that value in the corresponding
        // global stiffness matrix coordinate.
        // Example below for local stiffness matrix of element connecting nodes 0 and 3 in 3D (where a1, b1 and t1 are
        // x rotation of node 0, y rotation of node 0 and z rotation of node 0 respectively.
        // u0 v0 z0 a0 b0 t0 u3 v3 z3 a3 b3 t3
        // -  -  -  -  -  -  -  -  -  -  -  - u0
        // -  -  -  -  -  -  -  -  -  -  -  - v0
        // -  -  -  -  -  -  -  -  -  -  -  - z0
        // -  -  -  -  -  -  -  -  -  -  -  - a0
        // -  -  -  -  -  -  -  -  -  -  -  - b0
        // -  -  -  -  -  -  -  -  -  -  -  - t0
        // -  -  v  -  -  -  -  -  -  -  -  - u3
        // -  -  -  -  -  -  -  -  -  -  -  - v3
        // -  -  -  -  -  -  -  -  -  -  -  - z3
        // -  -  -  -  -  -  -  -  -  -  -  - a3
        // -  -  -  -  -  -  -  -  -  -  -  - b3
        // -  -  -  -  -  -  -  -  -  -  -  - t3
        // the local non-zero value 'v' will be placed in the column corresponding to z0 and the row corresponding to u3,
        // this means it should be in the column 2 of the global (or the 3rd column, where 0 is considered the first column)
        // as it is the 3rd column of node 0's DOF and it will be in the 18th row of the global matrix as it corresponds
        // to node3 which is (3*6 + 0) -> every node has 6 rows sequentially so node 3 will be after the first 18 columns
        // of nodes 0, 1 and 2. So following through with the code below:
        // v is a non-zero value so |k(row,col) != 0| is met
        // row < 6 in the local matrix meaning it corresponds to the first element and it follows the formula above so that
        // global_row = (6 * 0) + 2 = 2
        // col >=6 as it is in column 6 of the local and with el_n2 equalling element 3
        // global_col = (6 * 3) - 6 + 6 = 18
        // So for an element connecting node 0 and node 3, value v which is in the point above would be placed in (2,18) of
        // the global matrix.

        for (int col=0; col<12; col++) {
            for (int row=0; row<12; row++) {
                if (k(row,col) != 0) {
                    int global_row;
                    int global_col;

                    if (row < 6) {
                        global_row = (6*el_n1)+row;
                    } else if (row >= 6) {
                        global_row = ((6*el_n2)-6)+row;
                    }

                    if (col < 6) {
                        global_col = (6*el_n1)+col;
                    } else if (col >= 6) {
                        global_col = ((6*el_n2)-6)+col;
                    }
                    Triplet<double> triplet = {global_row,global_col,k(row,col)};
                    sparse_ijv.push_back(triplet);
                    bool fixed_col = fixed_DOF.find(global_col) != fixed_DOF.end();
                    bool fixed_row = fixed_DOF.find(global_row) != fixed_DOF.end();

                    if (!fixed_col && !fixed_row) {
                        int row_adj = global_row - adjust_indices[global_row];
                        int col_adj = global_col - adjust_indices[global_col];
                        Triplet<double> adjusted_triplet = {row_adj,col_adj,k(row,col)};
                        sparse_ijv_afterBC.push_back(adjusted_triplet);
                    } else {
                        sparse_ijv_fixed.push_back(triplet);
                    }
                }
            }
        }
    }
    global_matrix.setFromTriplets(sparse_ijv.begin(), sparse_ijv.end());
}


VectorXd solve(const std::unordered_set<int>& fixed_indices_hash, const vector<Triplet<double>>& sparse_vector,
           const vector<Triplet<double>>& sparse_vector_free, const vector<Triplet<double>>& sparse_vector_fixed, VectorXd& disp_full,
                                   const int gDOF) {

    int size_after_BC = gDOF - fixed_indices_hash.size();
    SparseMatrix<double> global_matrix_afterBC(size_after_BC,size_after_BC);
    global_matrix_afterBC.setFromTriplets(sparse_vector_free.begin(), sparse_vector_free.end());

    VectorXd force_vector_input = VectorXd::Zero(gDOF);

    std::ifstream force_file("forces.txt");

    double force;
    for (int i=0; i<gDOF; i++){
        force_file >> force;
        force_vector_input[i] = force;
    }
    force_file.close();

    VectorXd force_vector_afterBC = VectorXd::Zero(size_after_BC);

    int f_counter = 0;
    for (int i=0; i<force_vector_input.size(); i++) {
        bool found = fixed_indices_hash.find(i) != fixed_indices_hash.end();
        if (!found) {
            force_vector_afterBC[f_counter] = force_vector_input[i];
            f_counter++;
        }
    }

    VectorXd disp_vector(size_after_BC);
    std::cout << "Solving for nodal displacements..." << std::endl;
    ConjugateGradient<SparseMatrix<double> > solver;
    std::cout << "Computing sparse matrix..." << std::endl;
    solver.compute(global_matrix_afterBC);
    std::cout << "Solving sparse matrix..." << std::endl;
    disp_vector = solver.solve(force_vector_afterBC);

    int d_counter = 0;
    for (int i=0; i<disp_full.size(); i++) {
        if (disp_full(i) == 1) {
            disp_full(i) = disp_vector(d_counter);
            d_counter++;
        }
    }

    std::cout << "Solving for reaction forces..." << std::endl;
    VectorXd forces_vector_full(gDOF);
    SparseMatrix<double> global_matrix_fixed(gDOF,gDOF);

    global_matrix_fixed.setFromTriplets(sparse_vector_fixed.begin(), sparse_vector_fixed.end());

    forces_vector_full = global_matrix_fixed * disp_full;

    return forces_vector_full;
}


void save_to_vtk(const vector<node_struct>& nodes, const vector<vector<int>>& elements, VectorXd& displacements,
                 VectorXd& forces, const int gDOF) {

    std::cout << "Outputting data to .vtk file..." << std::endl;

    std::ofstream vtk_file("structure.vtk");

    vtk_file << "# vtk DataFile Version 4.0\nshape1\nASCII\n\nDATASET UNSTRUCTURED_GRID\nPOINTS " << gDOF/6 << " FLOAT\n";

    for (node_struct node : nodes) {
        vtk_file << node.x << " " << node.y << " " << node.z << "\n";
    }

    vtk_file << "\nCELLS " << elements.size() << " " << elements.size()*3 << "\n";

    for (int i=0; i<elements.size(); i++) {
        vtk_file << "2 " << elements[i][0] << " " << elements[i][1] << "\n";
    }

    vtk_file << "\nCELL_TYPES " << elements.size() << "\n";

    for (int i=0; i<elements.size(); i++) {
        vtk_file << "3\n";
    }

    vtk_file << "\nPOINT_DATA " << gDOF/6 << "\nVECTORS displacement FLOAT\n";

    for (int i=0; i<gDOF; i+=6) {
        vtk_file << displacements[i] << " " << displacements[i+1] << " " << displacements[i+2] << "\n";
    }

    vtk_file << "\nVECTORS force FLOAT\n";
    for (int i=0; i<gDOF; i+=6) {
        vtk_file << forces[i] << " " << forces[i+1] << " " << forces[i+2] << "\n";
    }
    vtk_file.close();
}