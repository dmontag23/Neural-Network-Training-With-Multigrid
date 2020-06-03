#include "gtest/gtest.h"
#include "relax.h"
#include "../test_helper.h"

// setup a fixture to use in the tests below
class RelaxTest : public testing::Test {
 
 protected:

  const unsigned int m1 = 2.0;
  const unsigned int m2 = 5.0;

  const phiFuncType mult_2 = [](const weightType& input) 
  {
    weightType output = input;
	for (int i = 0; i < input.size(); i++)
	{
		output[i] *= 2.0;
	}
	return output;
  };

  const phiFuncType mult_3 = [](const weightType& input) 
  {
    weightType output = input;
    for (int i = 0; i < input.size(); i++)
    {
        output[i] *= 3.0;
    }
    return output;
   };

  Relax relax1{m1, {mult_2}};
  Relax relax2{m2, {mult_3}};

  const unsigned int input_size = 50;
  weightType test_weights;
  listOfWeights input;
  listOfWeights rhs;

  // setup test input to be used
  void SetUp() override 
  {

    // initialize the weights that will be propogated using the phi functions above
    Matrix<double, 2, 3> test_weights_1;
    Matrix<double, 1, 1> test_weights_2;
	test_weights_1 << 4.5, 3, 2,
                      8, 1, -.5;
    test_weights_2 << -3;
    test_weights = {test_weights_1, test_weights_2};

	// initialize the list of weights for the relaxation tests
	weightType zeros = {MatrixXd::Zero(2,3), MatrixXd::Zero(1,1)};
	for (int i = 0; i < input_size; i++)
    {
        i % 3 == 0 ? input.push_back(test_weights) : input.push_back(zeros);
    }

	// initialize the rhs for relaxation tests
	weightType random_weights = {MatrixXd::Random(2,3), MatrixXd::Random(1,1)};
	for (int i = 0; i < input_size; i++)
	{
		i % 2 == 0 ? rhs.push_back(random_weights) : rhs.push_back(zeros);
	}

  }

};

listOfWeights cRelax(listOfWeights input, const listOfWeights& rhs, const unsigned int& m, const vector<phiFuncType>& phi)
{
    for (int i = m; i < input.size(); i+=m)
    {
        weightType new_weights = phi[(i-1) % phi.size()](input[i-1]);
        transform(new_weights.begin(), new_weights.end(), rhs[i].begin(), input[i].begin(), plus<MatrixXd>());
    }
    return input;
}

listOfWeights fRelax(listOfWeights input, const listOfWeights& rhs, const unsigned int& m, const vector<phiFuncType>& phi)
{
    for (int i = 0; i < input.size(); i+=m)
    {
        for (int j = i+1; j < i + m and j < input.size(); j++)
        {
            weightType new_weights = phi[(i-1) % phi.size()](input[j-1]);
            transform(new_weights.begin(), new_weights.end(), rhs[j].begin(), input[j].begin(), plus<MatrixXd>());
        }
    }
    return input;
}

void cRelaxTest(const Relax& relax, const listOfWeights& input, const listOfWeights& rhs, const unsigned int& m, const vector<phiFuncType>& phi)
{

    // construct the expected output
    listOfWeights expected_output = cRelax(input, rhs, m, phi);

    // check that the value of the cRelax function and the expected output are the same
    listOfWeights relaxed_output = relax.cRelax(input, rhs);
    testListOfVectors(relaxed_output, expected_output);

}

void fcfRelaxTest(const Relax& relax, const listOfWeights& input, const listOfWeights& rhs, const unsigned int& m, const vector<phiFuncType>& phi)
{

    // construct the expected output
    listOfWeights expected_output = fRelax(input, rhs, m, phi);
    expected_output = cRelax(expected_output, rhs, m, phi);
    expected_output = fRelax(expected_output, rhs, m, phi);

    // check that the value of the fcfRelax function and the expected output are the same
    listOfWeights relaxed_output = relax.fcfRelax(input, rhs);
    testListOfVectors(relaxed_output, expected_output);

}

void fRelaxTest(const Relax& relax, const listOfWeights& input, const listOfWeights& rhs, const unsigned int& m, const vector<phiFuncType>& phi)
{

    // construct the expected output
    listOfWeights expected_output = fRelax(input, rhs, m, phi);

    // check that the value of the fRelax function and the expected output are the same
    listOfWeights relaxed_output = relax.fRelax(input, rhs);
    testListOfVectors(relaxed_output, expected_output);

}

TEST_F(RelaxTest, CRelax_even){

    cRelaxTest(relax1, input, rhs, m1, {mult_2});

}

TEST_F(RelaxTest, CRelax_odd){

	cRelaxTest(relax2, input, rhs, m2, {mult_3});

}

TEST_F(RelaxTest, FCFRelax_even){

    fcfRelaxTest(relax1, input, rhs, m1, {mult_2});

}

TEST_F(RelaxTest, FCFRelax_odd){

    fcfRelaxTest(relax2, input, rhs, m2, {mult_3});

}

TEST_F(RelaxTest, FRelax_even){

    fRelaxTest(relax1, input, rhs, m1, {mult_2});

}

TEST_F(RelaxTest, FRelax_odd){

    fRelaxTest(relax2, input, rhs, m2, {mult_3});

}

TEST_F(RelaxTest, GetM) {

    unsigned int relax_m = relax1.getM();
    ASSERT_EQ(relax_m, m1) << "The m from the relax class " << relax_m << " is not equal to " << m1;

}

TEST_F(RelaxTest, GetPhi){

    // test that the output returned from both phi functions is the same
    weightType phi_output = relax1.getPhi()[0](test_weights);
    weightType mult_2_output = mult_2(test_weights);
    testVectors(phi_output, mult_2_output);

}

TEST_F(RelaxTest, SetM){
    
    // ensure the relax class is initialized properly
    unsigned int relax_m = relax1.getM();
    ASSERT_EQ(relax_m, m1) << "The m from the relax class " << relax_m << " is not equal to " << m1;
    
    // give the relax class a new m and test to make sure the new m is set correctly
    unsigned int new_m = 12;
    relax1.setM(new_m);
    relax_m = relax1.getM();
    ASSERT_EQ(relax_m, new_m) << "The new m from the relax class " << relax_m << " is not equal to " << new_m;
}

TEST_F(RelaxTest, SetPhi){

    // ensure the relax class is initialized properly
    weightType phi_output = relax1.getPhi()[0](test_weights);
    weightType mult_2_output = mult_2(test_weights);
    testVectors(phi_output, mult_2_output);

    // give the relax class a new phi and test to make sure the new phi is set correctly
	relax1.setPhi({mult_3});
    phi_output = relax1.getPhi()[0](test_weights);
    weightType mult_3_output = mult_3(test_weights);
    testVectors(phi_output, mult_3_output);

}