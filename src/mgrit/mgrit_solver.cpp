#include "mgrit_solver.h"

unsigned int MGRITSolver::getCoarseningFactor() const
{
   return m;
}

bool MGRITSolver::getDisplayStats() const
{
   return display_stats;
}

unsigned int MGRITSolver::getMaxLevel() const
{
   return max_level;
}

vector<vector<phiFuncType>> MGRITSolver::getPhis() const
{
   return phis;
}

listOfWeights MGRITSolver::fIteration(listOfWeights w0, const listOfWeights& rhs0)
{

   // apply the initial relaxation to the weights
   relaxer.setPhi(phis[current_level]);
   w0 = relaxer.fcfRelax(w0, rhs0);

   // calculate the residual on the fine level
   helper.setPhi(phis[current_level]);
   listOfWeights r0 = helper.residual(w0, rhs0);

   // restrict the weights and the residual to the coarse level
   mover.setCoarseningFactor(m);
   vector<listOfWeights> coarse_objects = mover.restrict({w0, r0});
   listOfWeights w1 = coarse_objects[0];  // get the coarse weights
   listOfWeights r1 = coarse_objects[1];  // get the coarse residual
   current_level++;
   
   // Construct the rhs of the linear system on the coarse level
   helper.setPhi(phis[current_level]);
   listOfWeights rhs1 = helper.addOrSubtract(helper.matMultiply(w1), r1, plus<MatrixXd>());

   // Solve the coarse linear system if on the coarsest level - otherwise recurse down to the next grid level
   listOfWeights v1 = (current_level >= (max_level-1) ? helper.forwardSolve(rhs1) : fIteration(w1, rhs1));

   // calculate the coarse level error approximation
   listOfWeights e1 = helper.addOrSubtract(v1, w1, minus<MatrixXd>());

   // project the weights from the coarse level to the fine level
   mover.setCoarseningFactor(m);
   w0 = mover.project(w0, e1);
   current_level--;

   // apply f relaxation to the weights
   relaxer.setPhi(phis[current_level]);
   w0 = relaxer.fRelax(w0, rhs0);

   // calculate the residual on the fine level
   helper.setPhi(phis[current_level]);
   r0 = helper.residual(w0, rhs0);

   // restrict the weights and the residual to the coarse level
   mover.setCoarseningFactor(m);
   coarse_objects = mover.restrict({w0, r0});
   w1 = coarse_objects[0];  // get the coarse weights
   r1 = coarse_objects[1];  // get the coarse residual
   current_level++;

   // Construct the rhs of the linear system on the coarse level
   helper.setPhi(phis[current_level]);
   rhs1 = helper.addOrSubtract(helper.matMultiply(w1), r1, plus<MatrixXd>());

   // Solve the coarse linear system if on the coarsest level - otherwise recurse down to the next grid level
   v1 = (current_level >= (max_level-1) ? helper.forwardSolve(rhs1) : vIteration(w1, rhs1));

   // calculate the coarse level error approximation
   e1 = helper.addOrSubtract(v1, w1, minus<MatrixXd>());

   // project the weights from the coarse level to the fine level
   mover.setCoarseningFactor(m);
   w0 = mover.project(w0, e1);
   current_level--;

   // apply f relaxation to the weights
   relaxer.setPhi(phis[current_level]);
   w0 = relaxer.fRelax(w0, rhs0);

   return w0;
}

MGRITHelper MGRITSolver::makeMGRITHelperObject(const vector<phiFuncType>& phi) const
{
   MGRITHelper helper(phi);
   return helper;
}

MoveGrids MGRITSolver::makeMoveGridsObject(const unsigned int& m) const
{
   MoveGrids mover(m);
   return mover;
}

Relax MGRITSolver::makeRelaxObject(const unsigned int& m, const vector<phiFuncType>& phi) const
{
   Relax relaxer(m, phi);
   return relaxer;
}

MGRITSolver::MGRITSolver(unsigned int my_m, vector<vector<phiFuncType>> my_phis, unsigned int my_max_level, bool my_display_stats) : m{my_m}, 
                                                                                                                             phis{my_phis}, 
                                                                                                                             max_level{my_max_level}, 
                                                                                                                             mover{makeMoveGridsObject(my_m)}, 
                                                                                                                             helper{makeMGRITHelperObject(my_phis[current_level])}, 
                                                                                                                             relaxer{makeRelaxObject(my_m, my_phis[current_level])},
                                                                                                                             display_stats{my_display_stats}{}

listOfWeights MGRITSolver::run(listOfWeights w0, const listOfWeights& rhs0, const double& tol, const bool& f_cycle)
{

   unsigned int iter_num = 0;  // initialize a counter to count the number of iterations MGRIT needs to converge

   // calculate the initial euclidean norm of the residual
   helper.setPhi(phis[current_level]);
   double r0_norm = helper.euclideanNorm(helper.residual(w0, rhs0)); 
   double residual_norm = r0_norm;

   // iterate until the euclidean norm of the residual is less than the desired tolerance
   while (residual_norm >= tol)
   {
      iter_num++;
      w0 = f_cycle? fIteration(w0, rhs0) : vIteration(w0, rhs0); // run v or f cycles depending on the user input

      // calculate the norm of the new residual on the fine level
      helper.setPhi(phis[current_level]);
      residual_norm = helper.euclideanNorm(helper.residual(w0, rhs0));

      // display stats about MGRIT as it is running if the flag is set
      if (display_stats)
      {
         cout << "Iteration Number: " << iter_num << endl;
         cout << "Euclidean Norm of the Residual: " << residual_norm << endl;
      }
   }

   // display the average convergence rate
   if (display_stats)
   {
      double rho = pow(residual_norm / r0_norm, 1.0 / iter_num);
      cout << "Average Convergence Rate: " << rho << endl;
   }
   
   return w0;
}

void MGRITSolver::setCoarseningFactor(unsigned int my_m)
{
   m = my_m;
}

void MGRITSolver::setDisplayStats(bool my_display_stats)
{
   display_stats = my_display_stats;
}

void MGRITSolver::setMaxLevel(unsigned int my_max_level)
{
   max_level = my_max_level;
}

void MGRITSolver::setPhis(vector<vector<phiFuncType>> my_phis)
{
   phis = my_phis;
}

listOfWeights MGRITSolver::vIteration(listOfWeights w0, const listOfWeights& rhs0)
{

   // apply the initial relaxation to the weights
   relaxer.setPhi(phis[current_level]);
   w0 = relaxer.fcfRelax(w0, rhs0);

   // calculate the residual on the fine level
   helper.setPhi(phis[current_level]);
   listOfWeights r0 = helper.residual(w0, rhs0);

   // restrict the weights and the residual to the coarse level
   mover.setCoarseningFactor(m);
   vector<listOfWeights> coarse_objects = mover.restrict({w0, r0});
   listOfWeights w1 = coarse_objects[0];  // get the coarse weights
   listOfWeights r1 = coarse_objects[1];  // get the coarse residual
   current_level++;
   
   // Construct the rhs of the linear system on the coarse level
   helper.setPhi(phis[current_level]);
   listOfWeights rhs1 = helper.addOrSubtract(helper.matMultiply(w1), r1, plus<MatrixXd>());

   // Solve the coarse linear system if on the coarsest level - otherwise recurse down to the next grid level
   listOfWeights v1 = (current_level >= (max_level-1) ? helper.forwardSolve(rhs1) : vIteration(w1, rhs1));

   // calculate the coarse level error approximation
   listOfWeights e1 = helper.addOrSubtract(v1, w1, minus<MatrixXd>());

   // project the weights from the coarse level to the fine level
   mover.setCoarseningFactor(m);
   w0 = mover.project(w0, e1);
   current_level--;

   // apply f relaxation to the weights on the fine level
   relaxer.setPhi(phis[current_level]);
   w0 = relaxer.fRelax(w0, rhs0);

   return w0;
}