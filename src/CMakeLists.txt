# add the executable
add_executable(three_layer_problem three_layer_problem.cpp mgrit/mgrit_helper.cpp mgrit/mgrit_solver.cpp mgrit/move_grids.cpp mgrit/relax.cpp neural_network/neural_network.cpp)
target_link_libraries(three_layer_problem Eigen3::Eigen)
add_executable(four_layer_problem four_layer_problem.cpp mgrit/mgrit_helper.cpp mgrit/mgrit_solver.cpp mgrit/move_grids.cpp mgrit/relax.cpp neural_network/neural_network.cpp)
target_link_libraries(four_layer_problem Eigen3::Eigen)