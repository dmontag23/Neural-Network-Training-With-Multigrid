# Parallelizing Neural Network Training with Multigrid
This repository contains the code that implements the MGRIT algorithm to train artificial neural networks as outlined in [this paper](docs/Parallelizing_Over_ANN_Training_Runs.pdf). 

## Requirements
In order to use this code, you must have a g++ compiler installed on your system along with the following packages:

- Cmake Version 3.10 or higher [found here](https://cmake.org/)
- Eigen Version 3.3 or higher [found here](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)
- Googletest Version 1.10 or higher [found here](https://github.com/google/googletest)

Note that, in order for the code to work correctly, these packages need to be installed in such a way that they are avaliable to all projects on the operating system. For example, it is recommended to install the Eigen package using Cmake as described in the INSTALL file provided by Eigen. Similarly, [this article](https://www.srcmake.com/home/google-cpp-test-framework) describes how to install the Googletest Framework in the libraries folder on Ubuntu, which ensures that it can be used by any project.

## Downloading and Compiling the Code
In order to use this code, clone or download the reposity into a directory of your choice. Create a *build* directory and then execute the following commands from that directory:

```
cmake ..
make
```

A *src* and a *tests* directory will be created. To ensure the installation was successful, run the unit tests as described in the section below.

## Running the Unit Tests and Project Executables
The following instructions assume the user has already built the project as described in the previous section and has navigated to their *build* (or equivalent) directory. To run the unit tests, navigate into the *tests* directory and run the *Multigrid_test* executable (e.g. using `./Multigrid_test`). This will run all of the unit tests in the project, which should pass. In order to run the project code, navigate into the *src* directory and run either the *three_layer_problem* or the *four_layer_problem* executable. Note that any changes in the source code will require the user to recompile the code using `make` before re-running the unit tests or project executables.

## Running Different Test Cases
In order to run different test cases corresponding to the paper, navigate to the *src* directory (in the home directory, not in the build directory). Open either the *three_layer_problem.cpp* or *four_layer_problem.cpp* file and scroll down to the *main* function. At the beginning of the function are the lines:

```
const int N = 100;                        // Number of training steps
const int m = 2;                          // Coarsening factor
const int max_level = 10;                 // The maximum level the MGRIT algorithm recurses to
const float alpha_b = 0.1;                // The learning rate of the neural network on the fine grid
const float alpha_max = 30.0;             // The maximum learning rate the algorithm can increase to on the coarse grids
const bool serialized_training = true;    // If true, the nn trains in a serial fashion, otherwise it trains in a batch fashion
const bool f_cycles = true;               // Determine whether to run F cycles (if false, the algorithm runs V cycles)
const bool display_output = true;         // Displays stats about the MGRIT algorithm as it is running
```

These are the parameters that need to be changed in order to run different test cases. For instance, the example above uses the MGRIT algorithm to train a neural network 100 times in a serialized manner with 10 grids. Each grid has half the number of nodes as the previous grid (since the coarsening factor is 2). The learning rate on the fine grid is 0.1, which doubles on each successive fine grid until it reaches or exceeds 30 and the algorithm uses F-cycles as its method of recursion (this example corresponds to Table 7 from [this report](docs/Multigrid_Project_Report.pdf)).

Once you have changed these parameters, save your changes and navigate to your build directory. Run `make` to recompile the code. Then run the new executable.