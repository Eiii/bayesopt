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

#include <ctime>
#include "bayesoptbase.hpp"

#include "log.hpp"
#include "posteriormodel.hpp"


namespace bayesopt
{


  BayesOptBase::BayesOptBase(size_t dim, bopt_params parameters):
    mParameters(parameters), mDims(dim)
  {
    // Setting verbose stuff (files, levels, etc.)
    int verbose = mParameters.verbose_level;
    if (verbose>=3)
      {
	FILE* log_fd = fopen( mParameters.log_filename , "w" );
	Output2FILE::Stream() = log_fd; 
	verbose -= 3;
      }

    switch(verbose)
      {
      case 0: FILELog::ReportingLevel() = logWARNING; break;
      case 1: FILELog::ReportingLevel() = logINFO; break;
      case 2: FILELog::ReportingLevel() = logDEBUG4; break;
      default:
	FILELog::ReportingLevel() = logERROR; break;
      }

    // Random seed
    if (mParameters.random_seed < 0) mParameters.random_seed = std::time(0); 
    mEngine.seed(mParameters.random_seed);

    // Posterior surrogate model
    mModel.reset(PosteriorModel::create(dim,parameters,mEngine));
    
    // Configure iteration parameters
    if (mParameters.n_init_samples <= 0)
      {
	mParameters.n_init_samples = 
	  static_cast<size_t>(ceil(0.1*mParameters.n_iterations));	
      }
  }

  BayesOptBase::~BayesOptBase()
  { } // Default destructor


  void BayesOptBase::stepOptimization()
  {
    // Find what is the next point.
    vectord xNext = nextPoint(); 
    stepOptimizationFixed(xNext);
  }

  void BayesOptBase::stepOptimizationFixed(const vectord& next)
  {
    vectord xNext = next;
    double yNext = evaluateSampleInternal(xNext);

    // If we are stuck in the same point for several iterations, try a random jump!
    if (mParameters.force_jump)
      {
	if (std::pow(mYPrev - yNext,2) < mParameters.noise)
	  {
	    mCounterStuck++;
	    FILE_LOG(logDEBUG) << "Stuck for "<< mCounterStuck << " steps";
	  }
	else
	  {
	    mCounterStuck = 0;
	  }
	mYPrev = yNext;

	if (mCounterStuck > mParameters.force_jump)
	  {
	    FILE_LOG(logINFO) << "Forced random query!";
	    xNext = samplePoint();
	    yNext = evaluateSampleInternal(xNext);
	    mCounterStuck = 0;
	  }
      }

    addSampleToModel(xNext, yNext);

    // Update surrogate model
    bool retrain = ((mParameters.n_iter_relearn > 0) && 
		    ((mCurrentIter + 1) % mParameters.n_iter_relearn == 0));

    if (retrain)  // Full update
      {
	mModel->updateHyperParameters();
	mModel->fitSurrogateModel();
      }
    else          // Incremental update
      {
	mModel->updateSurrogateModel();
      } 
    plotStepData(mCurrentIter,xNext,yNext);
    mModel->updateCriteria(xNext);
    mCurrentIter++;
  }

  // TODO -- ehhhh
  void BayesOptBase::addSample(vectord x, double y)
  {
    addSampleToModel(x, y);
    mModel->updateHyperParameters();
    mModel->updateSurrogateModel();
    mModel->updateCriteria(x);

    bool retrain = true; // TODO - How should adding an arbitrary, potentially 'useless' sample be handled?
    if (retrain)  // Full update
      {
        mModel->updateHyperParameters();
        mModel->fitSurrogateModel();
      }
  }

  bool BayesOptBase::removeSample(vectord x, double y)
  {
    int idx = -1;
    for (int i = 0; i < mXPoints.size(); i++) {
      //Make sure mXPoints == x, mYPoints == y
      bool vec_eq = true;
      for (int j = 0; j < mXPoints[i].size(); j++) { if (mXPoints[i](j) != x(j)) vec_eq = false; }

      if (vec_eq && mYPoints(i) == y) {
        idx = i;
      }
    }

    if (idx > 0) {
      mModel.reset(PosteriorModel::create(mDims,mParameters,mEngine));
      mModel->setSamples(mXPoints, mYPoints);
      mModel->fitSurrogateModel();
      bool retrain = true; // TODO - when should we retrain? When shouldn't we? Always?
      if (retrain) {
        mModel->updateHyperParameters();
        mModel->fitSurrogateModel();
      }
      return true;
    } else {
      return false;
    }
  }

  void BayesOptBase::stepBatchOptimization(int width)
  {
    // Find what is the next point.
    vecOfvec xNexts = nextBatchPoints(width);

    for(int i = 0; i < width; i++)
      {
        vectord xNext = xNexts[i];
        double yNext = evaluateSampleInternal(xNext);

        addSampleToModel(xNext, yNext);
        mModel->updateSurrogateModel();
        mModel->updateCriteria(xNext);

        plotStepData(mCurrentIter,xNext,yNext); //TODO -- Fix up stuff so this output makes sense
      }

    // Update surrogate model
    bool retrain = ((mParameters.n_iter_relearn > 0) && 
		    ((mCurrentIter + 1) % mParameters.n_iter_relearn == 0));

    if (retrain)  // Full update
      {
        mModel->updateHyperParameters();
        mModel->fitSurrogateModel();
      }

    mCurrentIter++;
  }

  vecOfvec BayesOptBase::nextBatchPoints(int width)
  {
    vecOfvec result;
    boost::scoped_ptr<PosteriorModel> selectionModel(mModel->clone(mEngine));

    for(size_t i = 0; i < width; i++)
      {
        selectionModel->fitSurrogateModel();

        vectord xNext = nextPoint(); 
        double yNext = selectionModel->getPrediction(xNext)->getMean();
        result.push_back(xNext);

        selectionModel->addSample(xNext, yNext);
        selectionModel->updateCriteria(xNext); //TODO - necessary?
      }

    return result;
  }

  void BayesOptBase::initializeOptimization()
  {
    size_t nSamples = mParameters.n_init_samples;

    matrixd xPoints(nSamples,mDims);
    vectord yPoints(nSamples);

    sampleInitialPoints(xPoints,yPoints);
    mModel->setSamples(xPoints,yPoints);
 
    if(mParameters.verbose_level > 0)
      {
	mModel->plotDataset(logDEBUG);
      }
    
    mModel->updateHyperParameters();
    mModel->fitSurrogateModel();
    mCurrentIter = 0;

	mCounterStuck = 0;
	mYPrev = 0.0;
  }


  void BayesOptBase::optimize(vectord &bestPoint)
  {
    initializeOptimization();
    assert(mDims == bestPoint.size());
    
    for (size_t ii = 0; ii < mParameters.n_iterations; ++ii)
      {      
	stepOptimization();
      }
   
    bestPoint = getFinalResult();
  } // optimize

  vectord BayesOptBase::nextPoint()
  {

    //Epsilon-Greedy exploration (see Bull 2011)
    if ((mParameters.epsilon > 0.0) && (mParameters.epsilon < 1.0))
      {
	randFloat drawSample(mEngine,realUniformDist(0,1));
	double result = drawSample();
	FILE_LOG(logINFO) << "Trying random jump with prob:" << result;
	if (mParameters.epsilon > result)
	  {
	    FILE_LOG(logINFO) << "Epsilon-greedy random query!";
	    return samplePoint();
	  }
      }

    vectord Xnext(mDims);    

    // GP-Hedge and related algorithms
    if (mModel->criteriaRequiresComparison())
      {
	bool changed = true;

	mModel->setFirstCriterium();
	while (changed)
	  {
	    findOptimal(Xnext);
	    changed = mModel->setNextCriterium(Xnext);
	  }
	std::string name = mModel->getBestCriteria(Xnext);
	FILE_LOG(logINFO) << name << " was selected.";
      }
    else  // Standard "Bayesian optimization"
      {
	FILE_LOG(logDEBUG) << "------ Optimizing criteria ------";
	findOptimal(Xnext);
      }
    return Xnext;
  }

  void BayesOptBase::addSampleToModel(vectord x, double y) {
    mModel->addSample(x,y);
    mXPoints.push_back(x);
    mYPoints.resize(mYPoints.size()+1);
    mYPoints(mYPoints.size()-1) = y;
  }


  // Potential inline functions. Moved here to simplify API and header
  // structure.
  double BayesOptBase::evaluateCriteria(const vectord& query)
  {
    if (checkReachability(query)) return mModel->evaluateCriteria(query);
    else return 0.0;
  }

  double BayesOptBase::evaluateCriteriaWithMin(const vectord& query, double altMin)
  {
    if (checkReachability(query)) return mModel->evaluateCriteriaWithMin(query, altMin);
    else return 0.0;
  }

  vectord BayesOptBase::getPointAtMinimum() 
  { return mModel->getPointAtMinimum(); };
  
  double BayesOptBase::getValueAtMinimum()
  { return mModel->getValueAtMinimum(); };

  ProbabilityDistribution* BayesOptBase::getPrediction(const vectord& query)
  { return mModel->getPrediction(query); };

   const Dataset* BayesOptBase::getData()
  { return mModel->getData(); };

  bopt_params* BayesOptBase::getParameters() 
  {return &mParameters;};


} //namespace bayesopt

