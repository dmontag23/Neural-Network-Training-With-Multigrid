#include "relax.h"

listOfWeights Relax::cRelax(listOfWeights weights, const listOfWeights& rhs) const
{
   for (int i = m; i < weights.size(); i += m)
   {
      weightType phi_of_w = phi[(i-1) % phi.size()](weights[i-1]);
      transform(phi_of_w.begin(), phi_of_w.end(), rhs[i].begin(), weights[i].begin(), plus<MatrixXd>());
   }
   
   return weights;
}

listOfWeights Relax::fcfRelax(listOfWeights weights, const listOfWeights& rhs) const
{
   weights = fRelax(weights, rhs);
   weights = cRelax(weights, rhs);
   weights = fRelax(weights, rhs);
   return weights;
}

listOfWeights Relax::fRelax(listOfWeights weights, const listOfWeights& rhs) const
{
   for (int i = 0; i < weights.size(); i += m)
   {
      for (int j = i+1; j < i+m and j < weights.size(); j++)
      {
         weightType phi_of_w = phi[(j-1) % phi.size()](weights[j-1]);
         transform(phi_of_w.begin(), phi_of_w.end(), rhs[j].begin(), weights[j].begin(), plus<MatrixXd>());
      }
   }

   return weights;
}

unsigned int Relax::getM() const
{
   return m;
}

vector<phiFuncType> Relax::getPhi() const
{
   return phi;
}

Relax::Relax(unsigned int my_m, vector<phiFuncType> my_phi) : m{my_m}, phi{my_phi}{}

void Relax::setM(unsigned int my_m)
{
   m = my_m;
}

void Relax::setPhi(vector<phiFuncType> my_phi)
{
   phi = my_phi;
}