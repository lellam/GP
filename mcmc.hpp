#ifndef MCMC_H
#define MCMC_H

#include <iostream>
#include <cmath>
#include <ctime>
#include <random>

#include "utility.hpp"

class RandomWalkMetropolis
{
public:
	RandomWalkMetropolis(double (*f)(const std::vector<double>&)) : target(f), uniform(0, 1), standard_normal(0,1) {}
	
	void run(std::vector<double> x, unsigned int mcmc_n, unsigned int burn, unsigned int thin, double step_size)
	{
		unsigned int dim = x.size();
		accept_n = 0;
		cached_target = target(x);
		clock_t t_start = clock();
	
		for(unsigned int i = 0; i < mcmc_n; ++i)
		{
			// Sample from rw
			std::vector<double> x_prop = x;
			for(unsigned int j = 0; j <dim; ++j)
				x_prop[j] += sqrt(step_size)*standard_normal(gen);
				
			// Compute acceptance probability
			double new_target = target(x_prop);
			double accept_prob = exp(new_target - cached_target);
			
			// Accept with probability accept_prob
			if(uniform(gen) < accept_prob)
			{
				x = std::move(x_prop);
				cached_target = new_target;
				++accept_n;
			}
			
			// Keep track of samples
			if(i >= burn and i%thin == 0)
				samples.push_back(x);
		}
		
		accept_ratio = float(accept_n)/float(mcmc_n);
		clock_t t_end = clock();
		double t_elapsed = double(t_end - t_start) / CLOCKS_PER_SEC;
	
		std::cout << "MCMC complete in " << t_elapsed << " with acceptance rate of " << accept_ratio << "." << std::endl;
	}
	
	std::vector<std::vector<double> > samples;
	double accept_ratio;
private:
	double (*target)(const std::vector<double>&);
	std::uniform_real_distribution<double> uniform;
	std::normal_distribution<double> standard_normal;
	double cached_target;
	unsigned int accept_n;
};

#endif