#ifndef HH_TEST_HELPER_HH
#define HH_TEST_HELPER_HH

#include <vector>

#include "gtest/gtest.h"

using namespace std;

template <typename T>
void testVectors(vector<T> actual_values, vector<T> expected_values)
{
	ASSERT_EQ(actual_values.size(), expected_values.size()) << "The actual values vector and the expected values vector are of unequal length";
	for (int i = 0; i < actual_values.size(); i++) 
	{
	    ASSERT_EQ(actual_values[i], expected_values[i]) << "The actual values vector and the expected values vector differ at index " << i;
	}
}

template <typename T>
void testListOfVectors(vector<vector<T>> actual_values, vector< vector<T>> expected_values)
{
	for (int i = 0; i < actual_values.size(); i++)
	{
		testVectors(actual_values[i], expected_values[i]); 
	}
}

#endif