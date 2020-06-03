#include "move_grids.h"

unsigned int MoveGrids::getCoarseningFactor() const
{
   return m;
}

MoveGrids::MoveGrids(unsigned int my_m) : m{my_m}{}

listOfWeights MoveGrids::project(listOfWeights w0, const listOfWeights& e1) const
{
   for (int i = 0; i < e1.size(); i++)
   {
      transform(w0[i*m].begin(), w0[i*m].end(), e1[i].begin(), w0[i*m].begin(), plus<MatrixXd>());
   }
   return w0;
}

vector<listOfWeights> MoveGrids::restrict(const vector<listOfWeights>& fine_grid_objects) const
{
   int num_of_objects = fine_grid_objects.size();
   int num_coarse_nodes = ceil(static_cast<float>(fine_grid_objects[0].size())/m);

   vector<listOfWeights> new_coarse_objects;
   for (int i = 0; i < num_of_objects; i++)
   {
      listOfWeights coarse_object;
      for (int j = 0; j < num_coarse_nodes; j++)
      {
         coarse_object.push_back(fine_grid_objects[i][j * m]);
      }
      new_coarse_objects.push_back(coarse_object);
   }

   return new_coarse_objects;
}

void MoveGrids::setCoarseningFactor(unsigned int my_m)
{
   m = my_m;
}