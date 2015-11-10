/*
-------------------------------------------------------------------------
   This file is part of BayesOpt, an efficient C++ library for 
   Bayesian optimization.

   Copyright (C) 2011-2014 Ruben Martinez-Cantin <rmcantin@unizar.es>
 
   BayesOpt is free software: you can redistribute it and/or modify it 
   under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   BayesOpt is distributed in the hope that it will be useful, but 
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with BayesOpt.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
*/

#ifndef  _KERNEL_GAUSSIAN_HPP_
#define  _KERNEL_GAUSSIAN_HPP_

#include "kernels/kernel_atomic.hpp"

namespace bayesopt
{
  
  /**\addtogroup KernelFunctions */
  //@{

  /** \brief Square exponential (Gaussian) kernel. Isotropic version. */
  class SEIso: public ISOkernel
  {
  public:
    void init(size_t input_dim)
    { n_params = 1; n_inputs = input_dim;  };

    double operator()( const vectord &x1, const vectord &x2)
    {
      double rl = computeWeightedNorm2(x1,x2);
      double k = rl*rl;
      return exp(-k/2);
    };

    double gradient(const vectord &x1, const vectord &x2,
		    size_t component)
    {
      double rl = computeWeightedNorm2(x1,x2);
      double k = rl*rl;
      return exp(-k/2)*k;
    };
  };

  /** \brief Square exponential (Gaussian) kernel. Fixed-length isotropic version. */
  class SEIso_fixed: public ISOkernel
  {
  public:
    void init(size_t input_dim)
    { n_params = 0; n_inputs = input_dim;  };

    inline double computeFixedNorm2(const vectord &x1, const vectord &x2)
    {  
      assert(n_inputs == x1.size());
      assert(x1.size() == x2.size());
      double length = 0.05;
      return norm_2(x1-x2)/length; 
    };

    double operator()( const vectord &x1, const vectord &x2)
    {
      double rl = computeFixedNorm2(x1,x2);
      double k = rl*rl;
      return exp(-k/2);
    };

    double gradient(const vectord &x1, const vectord &x2,
		    size_t component)
    {
      double rl = computeFixedNorm2(x1,x2);
      double k = rl*rl;
      return exp(-k/2)*k;
    };
  };

  /** \brief Square exponential (Gaussian) kernel. ARD version. */
  class SEArd: public ARDkernel
  {
  public:
    void init(size_t input_dim)
    { n_params = input_dim;  n_inputs = input_dim; };

    double operator()( const vectord &x1, const vectord &x2 )
    {
      double rl = computeWeightedNorm2(x1,x2);
      double k = rl*rl;
      return exp(-k/2);
    };
  
    double gradient(const vectord &x1, const vectord &x2,
		    size_t component)
    {
      double rl = computeWeightedNorm2(x1,x2);
      double k = rl*rl;
      double r = (x1(component) - x2(component))/params(component);
      return exp(-k/2)*r*r;
    };
  };


  //@}

} //namespace bayesopt

#endif
