// MomentMatching.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"


#include "SPN.h"
#include <iostream>
#include <time.h>
#include <string>
#include <math.h>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <numeric>

using namespace std;
using Eigen::MatrixXd;

int main(int argc, char* argv[])
{


	srand((unsigned int)std::time(0));
	

	string command = argv[1];

    // command format
    // BMM RandomStructureAndMM <number_of_variables> <depth_of_random_structure> <max_number_of_children_per_sum_node> <path_to_train> <path_to_test>
	if (command == "RandomStructureAndMM") { // RandomStructureAndMM num_variables 
		int num_variables = std::stoi(argv[2]);
		int Depth = std::stoi(argv[3]);
		int sum_max_children = std::stoi(argv[4]);
		SPN newSPN(num_variables, 2);
		newSPN.generateRandomStructure(Depth, sum_max_children);
		newSPN.randomizeWeights();
		newSPN.normalizeWeights();

		string train_data = argv[5];
		string test_data = argv[6];
        
        newSPN.readtrainData(train_data);
        newSPN.readtestData(test_data);

		newSPN.doInference(newSPN.testData, test_data + "ll.out");
        
        for (int i = 0; i < 5; i++) {
            time_t now = time(0);
            newSPN.doMomentMatching(newSPN.trainData);
            time_t after = time(0);
            cout << "Time for doing moment matching: " << after - now << " seconds." << endl;
            newSPN.normalizeWeights();
            newSPN.doInference(newSPN.testData, test_data + "ll.out");
        }
		return 0;
	}



    // command format
    // BMM Multivariate_GMM <number_of_variables> <num_of_mixtures> <path_to_train> <path_to_test>
	if (command == "Multivariate_GMM") {

		int num_variables = std::stoi(argv[2]);
		int num_mixtures = std::stoi(argv[3]);
		SPN newSPN(num_variables, 1);

		string train_data = argv[4];
		newSPN.readtrainDataCont(train_data);

		string test_data = argv[5];
		newSPN.readtestDataCont(test_data);

		newSPN.generateContinuousMixtureMultivariate(num_mixtures);
		newSPN.randomizeWeightsContinuousMultivariate();

		// string spn_file = argv[6];
        // newSPN.writeSPNContinuousMultivariate(spn_file);

		newSPN.doInferenceContinuousMultivariate(newSPN.trainDataCont, "loglikelihoods.train");
		newSPN.doInferenceContinuousMultivariate(newSPN.testDataCont, "loglikelihoods.test");

		newSPN.doMomentMatchingContinuousMultivariate(newSPN.trainDataCont);
		//newSPN.doMomentMatchingContinuousMultivariate(newSPN.trainDataCont);

		newSPN.doInferenceContinuousMultivariate(newSPN.trainDataCont, "loglikelihoods.train");
		newSPN.doInferenceContinuousMultivariate(newSPN.testDataCont, "loglikelihoods.test");
	}

    // command format
    // BMM Multivariate_SPN <number_of_variables> <depth_of_spn> <max_number_of_children_for_sum_nodes> <number_of_variables_at_leaf_nodes> <path_to_train> <path_to_test>
    
	if (command == "Multivariate_SPN") {

		int num_variables = std::stoi(argv[2]);
		int Depth = std::stoi(argv[3]);
		int sum_max_children = std::stoi(argv[4]);
		int leaf_n_variables = std::stoi(argv[5]);
		SPN newSPN(num_variables, 1);

		string train_data = argv[6];
		newSPN.readtrainDataCont(train_data);

		string test_data = argv[7];
		newSPN.readtestDataCont(test_data);

		newSPN.generateContinuousRandomStructureMultivariate(Depth, sum_max_children, leaf_n_variables);
		newSPN.randomizeWeightsContinuousMultivariate();

		string spn_file = argv[8];
        
		newSPN.writeSPNContinuousMultivariate(spn_file);
		newSPN.doInferenceContinuousMultivariate(newSPN.trainDataCont, "loglikelihoods.train");
		newSPN.doInferenceContinuousMultivariate(newSPN.testDataCont, "loglikelihoods.test");
		newSPN.doMomentMatchingContinuousMultivariate(newSPN.trainDataCont);
		newSPN.doInferenceContinuousMultivariate(newSPN.trainDataCont, "loglikelihoods.train");
		newSPN.doInferenceContinuousMultivariate(newSPN.testDataCont, "loglikelihoods.test");
	}


	return 0;
}

