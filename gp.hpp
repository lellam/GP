#ifndef GP_H
#define GP_H

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <assert.h>
#include <cmath>
#include <random>

#include "utility.hpp"

class MVN
{
public:
	MVN(Eigen::VectorXd& m, Eigen::MatrixXd& c) : mean(m), 
												  covariance(c),
												  dim(m.size()),
												  standard_normal(0, 1)
	{
		assert(c.rows() == dim && c.cols() == dim);
		llt = Eigen::LLT<Eigen::MatrixXd>(covariance);
		l_cholesky =llt.matrixL();
		
		double root_log_det = 0.;
		for(unsigned int i = 0; i < dim; ++i)
			root_log_det += log(l_cholesky(i, i));
		norm = -0.5*dim*log(2*M_PI) - root_log_det;
		
		return;
	}
	
	double log_pdf(const Eigen::VectorXd& x)
	{
		assert(x.size() == dim);
		
		// Precomputed normalization term
		double ret = norm;
	
		//Compute exponent term y^T*z where Cz = y, y = (x-m)
		Eigen::VectorXd y = x - mean;
		Eigen::VectorXd z = llt.solve(y);
		ret += -0.5*y.transpose()*z;
		
		return ret;
	}
	
	Eigen::VectorXd rv()
	{
		// Draw z ~ N(0,1) and return Lz + m
		Eigen::VectorXd z(dim);
		for(unsigned int i = 0; i < dim; ++i)
			z(i) = standard_normal(gen);
			
		Eigen::VectorXd x = l_cholesky*z + mean;
		return x;
	}
	
private:
	Eigen::VectorXd& mean;
	Eigen::MatrixXd& covariance;
	Eigen::LLT<Eigen::MatrixXd> llt;
	Eigen::MatrixXd l_cholesky;
	double norm;
	unsigned int dim;
	std::normal_distribution<double> standard_normal;
};



class Kernel
{
public:
	Kernel(const Eigen::VectorXd& s, const std::vector<double>& p) : x_space(s), param(p), dim(s.size()) {}
	Eigen::MatrixXd operator()() const
	{
		Eigen::MatrixXd covariance(dim, dim);
		for(unsigned int i = 0; i < dim; ++i)
		{
			// Diagonal - last term fixes numerical stability issues
			covariance(i,i) = covariance_function(x_space(i), x_space(i)) + 10e-8;
			
			for(unsigned int j = i+1; j < dim; ++j)
			{
					covariance(i,j) = covariance_function(x_space(i), x_space(j));
					covariance(j,i) = covariance(i,j);
			}
		}
		return covariance;
	}

private:
	double covariance_function(const double x, const double y) const
	{
		// Exponential squared covariance function
		double l = param[0];
		assert (l > 0);
		double z = (x - y)/l;
		double z2 = z*z;
		double ret = exp(-0.5*z2);
		return ret;			
	}
	
	const Eigen::VectorXd& x_space;
	const std::vector<double>& param;
	unsigned int dim;
};


#endif