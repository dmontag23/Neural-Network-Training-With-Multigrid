#ifndef HH_MGRIT_SOLVER_HH
#define HH_MGRIT_SOLVER_HH

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "mgrit_helper.h"
#include "move_grids.h"
#include "relax.h"
#include "typedefs.h"

using namespace Eigen;
using namespace std;

class MGRITSolver {

   private:

      unsigned int current_level = 0;           // keeps track of the current grid level MGRIT is on
      bool display_stats;                       // displays stats about the MGRIT algorithm as it is running
      unsigned int m;                           // coarsening factor
      unsigned int max_level;                   // denotes the maximum coarse level grid MGRIT will recurse to
      vector<vector<phiFuncType>> phis;         // list of phi functions on each grid level
      
      // helper classes that assist in implementing the MGRIT algorithm
      MGRITHelper helper;
      MoveGrids mover;
      Relax relaxer;  

      listOfWeights fIteration(listOfWeights w0, const listOfWeights& rhs0);  
      MGRITHelper makeMGRITHelperObject(const vector<phiFuncType>& phi) const;
      MoveGrids makeMoveGridsObject(const unsigned int& m) const;
      Relax makeRelaxObject(const unsigned int& m, const vector<phiFuncType>& phi) const;   
      listOfWeights vIteration(listOfWeights w0, const listOfWeights& rhs0);

   public:

      MGRITSolver(unsigned int my_m, vector<vector<phiFuncType>> my_phis, unsigned int my_max_level, bool my_display_stats = false);

      // getters and setters
      unsigned int getCoarseningFactor() const;
      bool getDisplayStats() const;
      unsigned int getMaxLevel() const;
      vector<vector<phiFuncType>> getPhis() const;
      void setCoarseningFactor(unsigned int my_m);
      void setDisplayStats(bool my_display_stats);
      void setMaxLevel(unsigned int my_max_level);
      void setPhis(vector<vector<phiFuncType>> my_phis);

      listOfWeights run(listOfWeights w0, const listOfWeights& rhs0, const double& tol, const bool& f_cycle);
};

#endif