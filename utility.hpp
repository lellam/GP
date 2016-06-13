#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <random>

// Global PRNG
std::random_device rd;
std::mt19937_64 gen(rd());

void write_vec1d(const std::string& file_name, const std::vector<double>& vec)
{
	std::ofstream file;
	file.open(file_name.c_str()); 
	file << std::fixed;
	file.precision(std::numeric_limits<double>::digits10);
	for(auto e:vec)
		file << e << "\n";
	return;
}

void write_vec2d(const std::string& file_name, const std::vector<std::vector<double> >& vec)
{

	std::ofstream file;
	file.open(file_name.c_str()); 
	file << std::fixed;
	file.precision(std::numeric_limits<double>::digits10);
	for(auto v:vec)
	{
		for(auto e:v)
			file << e << " ";
		file << "\n";
	}

	return;
}

// Read/write vector routines
std::vector<std::vector<double> > read_vec2d(const std::string& file_name)
{
	std::vector<std::vector<double> > ret;
	
    std::ifstream file;
    file.open(file_name.c_str()); 
    std::string line;
    
    if(file.is_open())
    {
        while(file.good())
        {
            std::getline(file, line);
            
            std::vector<double> temp;
            std::stringstream ss(line);
            double d;
            
            while(ss >> d)
                temp.push_back(d);
            
            if(!temp.empty())
                ret.push_back(temp);
        }
        file.close();
    }
    return ret;
}

#endif