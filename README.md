# Parallelizing Neural Network Training with Multigrid
This repository contains the code that implements the MGRIT algorithm to train artificial neural networks as outlined in [this paper](https://arxiv.org/abs/1708.02276). 

## Requirements
In order to use this code, you must have a g++ compiler installed on your system, along with the following packages:

- Cmake Version 3.10 or higher [found here](https://cmake.org/)
- Eigen Version 3.3 or higher [found here](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)
- Googletest Version 1.10 or higher [found here](https://github.com/google/googletest)

## Downloading and Compiling the Code
In order to use this code, clone the reposity into a directory of your choice. Create a *build* directory and then execute the following commands from that directory:

'''
cmake ..
make
'''

A *src* and a *tests* directory will be created. To ensure the installation was successful, change into the *tests* directory and run the *Multigrid_test* executable. This will run all of the unit tests in the project, which should pass. In order to run the project code, change into the *src* directory and run either the *three_layer_problem* or the *four_layer_problem* executable.

## Running Different Test Cases
In order to run different test cases corresponding to the paper, navigate to the *src* directory (in the home directory, not the build directory). Open either the *three_layer_problem.cpp* or *four_layer_problem.cpp* file and scroll down to the *main* function. At the beginning of the function are the lines:

'''
const int N = 100;                        // Number of training steps
const int m = 2;                          // Coarsening factor
const int max_level = 10;                 // The maximum level the MGRIT algorithm recurses to
const float alpha_b = 0.1;                // The learning rate of the neural network on the fine grid
const float alpha_max = 30.0;             // The maximum learning rate the algorithm can increase to on the coarse grids
const bool serialized_training = true;    // If true, the nn trains in a serial fashion, otherwise it trains in a batch fashion
const bool f_cycles = true;               // Determine whether to run F cycles (if false, the algorithm runs V cycles)
const bool display_output = true;         // Displays stats about the MGRIT algorithm as it is running
'''

These are the parameters that need to be changed in order to run different test cases. For instance, the example uses the MGRIT algorithm to train a neural network 100 times in a serialized manner with 10 grids. Each grid has half the number of nodes as the previous grid (since the coarsening factor is 2). The learning rate on the fine grid is 0.1, which doubles on each successive fine grid until it reaches or exceeds 30 and the algorithm uses F-cycles as its method of recursion (for more details see THIS REPORT COME BACK AND ADD LINK!!!!!!!!!!!).

Once you have changed these parameters, navigate to your build directory and run 'make' to recompile the code. Then run the new executable.