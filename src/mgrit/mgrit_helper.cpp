#include "mgrit_helper.h"

double MGRITHelper::euclideanNorm(const listOfWeights& weights) const
{
   double norm = 0;
   for (int i = 0; i < weights.size(); i++)
   {
      for (MatrixXd weight : weights[i])
      {
         norm += weight.squaredNorm();
      }
   }
   norm = sqrt(norm);
   return norm;
}

listOfWeights MGRITHelper::forwardSolve(const listOfWeights& rhs) const
{
   listOfWeights result(rhs.size());
   result[0] = rhs[0];
   for (int i = 1; i < rhs.size(); i++)
   {
    transform(rhs[i].begin(), rhs[i].end(), phi[(i-1) % phi.size()](result[i-1]).begin(), back_inserter(result[i]), plus<MatrixXd>());
   }
   return result;
}

vector<phiFuncType> MGRITHelper::getPhi() const
{
   return phi;
}

listOfWeights MGRITHelper::matMultiply(const listOfWeights& weights) const
{
   listOfWeights result(weights.size());
   result[0] = weights[0];
   for (int i = 1; i < weights.size(); i++)
   {
      transform(weights[i].begin(), weights[i].end(), phi[(i-1) % phi.size()](weights[i-1]).begin(), back_inserter(result[i]), minus<MatrixXd>());
   }
   return result;
}

MGRITHelper::MGRITHelper(vector<phiFuncType> my_phi) : phi{my_phi}{}

listOfWeights MGRITHelper::residual(const listOfWeights& weights, const listOfWeights& rhs) const
{
   return addOrSubtract(rhs, matMultiply(weights), minus<MatrixXd>());
}

void MGRITHelper::setPhi(vector<phiFuncType> my_phi)
{
   phi = my_phi;
}