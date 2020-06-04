#ifndef HH_MGRIT_HELPER_HH
#define HH_MGRIT_HELPER_HH

#include <Eigen/Dense>
#include <vector>

#include "typedefs.h"

using namespace Eigen;
using namespace std;

class MGRITHelper {

   private:

      vector<phiFuncType> phi;        // phi functions used in the forward solve and the matrix multiplication

   public:

      MGRITHelper(vector<phiFuncType> my_phi);

      // getters and setters
      vector<phiFuncType> getPhi() const;
      void setPhi(vector<phiFuncType> my_phi);

      // template function must be implemented in the header
      template <typename T> 
      listOfWeights addOrSubtract(const listOfWeights& first_list, const listOfWeights& second_list, const T& add_or_subtract) const
      {
        listOfWeights result(first_list.size());
        transform(first_list.begin(), first_list.end(), second_list.begin(), result.begin(), 
                  [add_or_subtract](weightType w1, weightType w2) 
                  {
                    transform(w1.begin(), w1.end(), w2.begin(), w1.begin(), add_or_subtract);
                    return w1;
                  });

        return result;
      }

      double euclideanNorm(const listOfWeights& weights) const;
      listOfWeights forwardSolve(const listOfWeights& rhs) const;
      listOfWeights matMultiply(const listOfWeights& weights) const;
      listOfWeights residual(const listOfWeights& weights, const listOfWeights& rhs) const;

};

#endif