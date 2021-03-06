#ifndef HH_MOVE_GRIDS_HH
#define HH_MOVE_GRIDS_HH

#include <Eigen/Dense>
#include <vector>

#include "typedefs.h"

using namespace Eigen;
using namespace std;

class MoveGrids {

   private:

      unsigned int m;   // coarsening factor

   public:        

      MoveGrids(unsigned int my_m);

      // getters and setters
      unsigned int getCoarseningFactor() const;
      void setCoarseningFactor(unsigned int my_m);

      listOfWeights project(listOfWeights w0, const listOfWeights& e1) const;
      vector<listOfWeights> restrict(const vector<listOfWeights>& fine_grid_objects) const;

};

#endif