#ifndef HH_RELAX_HH
#define HH_RELAX_HH

#include <Eigen/Dense>
#include <vector>

#include "typedefs.h"

using namespace Eigen;
using namespace std;

class Relax {

   private:

      unsigned int m;           // coarsening factor
      vector<phiFuncType> phi;  // phi functions used to do the relaxation

   public:

      Relax(unsigned int my_m, vector<phiFuncType> my_phi);

      // getters and setters
      unsigned int getM() const;
      vector<phiFuncType> getPhi() const;
      void setPhi(vector<phiFuncType> my_phi);
      void setM(unsigned int my_m);

      listOfWeights cRelax(listOfWeights weights, const listOfWeights& rhs) const;
      listOfWeights fcfRelax(listOfWeights weights, const listOfWeights& rhs) const;
      listOfWeights fRelax(listOfWeights weights, const listOfWeights& rhs) const;

};

#endif