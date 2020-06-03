# fixes an error so that you can link with the pthread-library of Google Tests correctly
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(TEST_NAME ${CMAKE_PROJECT_NAME}_test)

# find the external Google Test library
find_package(GTest REQUIRED)

# add the executable
add_executable(${TEST_NAME} main.cpp test_helper.h mgrit/mgrit_helper_test.cpp mgrit/mgrit_solver_test.cpp mgrit/move_grids_test.cpp mgrit/relax_test.cpp neural_network/neural_network_test.cpp ${CMAKE_SOURCE_DIR}/src/mgrit/mgrit_helper.cpp ${CMAKE_SOURCE_DIR}/src/mgrit/mgrit_solver.cpp ${CMAKE_SOURCE_DIR}/src/mgrit/move_grids.cpp ${CMAKE_SOURCE_DIR}/src/mgrit/relax.cpp ${CMAKE_SOURCE_DIR}/src/neural_network/neural_network.cpp)
target_link_libraries(${TEST_NAME} Eigen3::Eigen gtest)