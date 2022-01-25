// Gmsh project created on Tue Oct 13 06:02:07 2020
SetFactory("OpenCASCADE");

head_radius = 87.53;
brain_ratio = 0.83;
csf_ratio = 0.023;
skull_ratio = 0.085;
scalp_ratio = 0.05;

//Sphere(1) = {0, 0, 0, head_radius};
//Sphere(2) = {0, 0, 0, head_radius*(brain_ratio + csf_ratio + skull_ratio + scalp_ratio)};
//Sphere(3) = {0, 0, 0, head_radius*(brain_ratio + csf_ratio + skull_ratio)};
//Sphere(4) = {0, 0, 0, head_radius*(brain_ratio + csf_ratio)};
Sphere(5) = {0, 0, 0, head_radius*(brain_ratio)};

//Physical Volume("Outer Boundary", 1) = {1};

/*
Physical Volume("Skin", 2) = {2};
Physical Volume("Skull", 3) = {3};
Physical Volume("CSF", 4) = {4};
Physical Volume("Brain", 5) = {5};

Physical Surface("Outer Boundary", 1) = {1};
Physical Surface("Skin", 2) = {2};
Physical Surface("Skull", 3) = {3};
Physical Surface("CSF", 4) = {4};
Physical Surface("Brain", 5) = {5};
*/

Mesh 2;
RefineMesh;
RefineMesh;
RefineMesh;
Mesh 3;
RefineMesh;


