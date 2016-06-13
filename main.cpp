#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

#include "utility.hpp"
#include "gp.hpp"
#include "mcmc.hpp"


// Global variables
Eigen::VectorXd x_data;
Eigen::VectorXd y_data;
unsigned int dim;


// Function for log-posterior density - this is the target density
double log_posterior (const std::vector<double>& log_param)
{
		// Construct kernel
		std::vector<double> param = {exp(log_param[0])};
		Kernel k(x_data, param);
		
		// Compute likelihood function
		Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
		Eigen::MatrixXd covariance = k();
		MVN gp(mean, covariance);
		double log_likelihood = gp.log_pdf(y_data);
		
		// Compute prior
		double log_prior = -0.5*log_param[0]*log_param[0]/10000.;
		
		// Return log posterior
		return log_likelihood + log_prior;
}


int main()
{
	/*
	// Uncomment this section to generate a new data set
	{
		// Key parameters
		dim = 100;
		std::vector<double> param = {.1};
	
		// Generate x-space
		x_data.resize(dim);
		double h = (1.-0.)/float(dim-1);
		for(unsigned int i = 0; i < dim; ++i)
			x_data(i) = i*h;
	
		// Build GP
		Kernel k(x_data, param);
		Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
		Eigen::MatrixXd covariance = k();
	
		// Sample from GP and save down
		MVN gp(mean, covariance);
		y_data = gp.rv();
	
		std::vector<std::vector<double> > data(dim, std::vector<double>(2));
		for(unsigned int i = 0; i < dim; ++i)
		{
			data[i][0] = x_data(i);
			data[i][1] = y_data(i);
		}

		write_vec2d("output/data.dat", data);
	}
	*/
	
	
	// Read in data file and update global vars.
	{
		std::vector<std::vector<double> > data = read_vec2d("output/data.dat");
		dim = data.size();
		x_data.resize(dim);
		y_data.resize(dim);
		
		for(unsigned int i = 0; i < dim; ++i)
		{
			x_data(i) = data[i][0];
			y_data(i) = data[i][1];
		}
	}
	
	
	// Run sampler
	RandomWalkMetropolis sampler(log_posterior);
	std::vector<double> init = {0.};
	sampler.run(init, 10000, 1000, 50, 0.005);


	// Save down results
	write_vec2d("output/out.dat", sampler.samples);
	
	return 0;
}