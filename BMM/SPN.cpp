/*
 * SPN.cpp
 *
 *  Created on: Jun 20, 2015
 *      Author: abdoo_000
 */
#define _USE_MATH_DEFINES
//#include "StdAfx.h"
#include "SPN.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <numeric>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <float.h>
#include <limits.h>
#include <assert.h> 
#include <string>
#include <queue>
#include <functional>
#include <random>
#include <chrono>
#include <map>
#include <Eigen/Eigenvalues> 



using namespace std;

#ifndef isnan 
#define isnan(x) ((x)!=(x)) 
#endif

void SPN::writeSPN(std::string file_name) {
	ofstream output_file(file_name);
	output_file << "##NODES##\n";
	for(vector<spnNode*>::iterator iter=spnNetwork.begin(); iter<spnNetwork.end(); iter++){
		spnNode* current = (*iter);
		if (current->Type == "Var") {
			continue;
		} else if (current->pChildren[0]->Type == "Var" ) {
			// print leave :)
			output_file << current->ID << ",BINNODE," << current->pChildren[0]->featureIdx << "," << current->weights[1] << "," << current->weights[0] << endl;
			
		} else if (current->Type == "Sum") {
			// print SUM
			output_file << current->ID << ",SUM\n";
		} else {
			// print PRD
			output_file << current->ID << ",PRD\n";
		}
	}
	output_file << "##EDGES##\n";

	for(vector<spnNode*>::iterator iter=spnNetwork.begin(); iter<spnNetwork.end(); iter++){
		spnNode* current = (*iter);
		if (current->Type == "Var") {
			continue;
		} else if (current->pChildren[0]->Type == "Var" ) {
			// print leave :)
			continue;
		} else if (current->Type == "Sum") {
			// print SUM
			for (int i = 0; i < current->children.size(); i++) {
				output_file << current->ID << "," << current->children[i] << "," << current->weights[i] << endl;
			}
		} else {
			// print PRD
			for (int i = 0; i < current->children.size(); i++) {
				output_file << current->ID << "," << current->children[i] << endl;
			}
		}
	}
	output_file.close();
}

void SPN::writeSPNContinuous(std::string file_name) {
	ofstream output_file(file_name);
	output_file << "##NODES##\n";
	for (vector<spnNode*>::iterator iter = spnNetwork.begin(); iter<spnNetwork.end(); iter++) {
		spnNode* current = (*iter);
		if (current->Type == "Var") {
			continue;
		}
		else if (current->pChildren[0]->Type == "Var") {
			// print leave :)
			output_file << current->ID << ",LEAVE," << current->pChildren[0]->featureIdx << "," << current->gaussian_mixture[0].NG[0] << "," << current->gaussian_mixture[0].NG[3]/ current->gaussian_mixture[0].NG[2] << endl;

		}
		else if (current->Type == "Sum") {
			// print SUM
			output_file << current->ID << ",SUM\n";
		}
		else {
			// print PRD
			output_file << current->ID << ",PRD\n";
		}
	}
	output_file << "##EDGES##\n";

	for (vector<spnNode*>::iterator iter = spnNetwork.begin(); iter<spnNetwork.end(); iter++) {
		spnNode* current = (*iter);
		if (current->Type == "Var") {
			continue;
		}
		else if (current->pChildren[0]->Type == "Var") {
			// print leave :)
			continue;
		}
		else if (current->Type == "Sum") {
			// print SUM
			for (int i = 0; i < current->children.size(); i++) {
				output_file << current->ID << "," << current->children[i] << "," << current->weights[i] << endl;
			}
		}
		else {
			// print PRD
			for (int i = 0; i < current->children.size(); i++) {
				output_file << current->ID << "," << current->children[i] << endl;
			}
		}
	}
	output_file.close();
}

void SPN::writeSPNContinuousMultivariate(std::string file_name) {
	ofstream output_file(file_name);
	output_file << "##NODES##\n";
	for (vector<spnNode*>::iterator iter = spnNetwork.begin(); iter<spnNetwork.end(); iter++) {
		spnNode* current = (*iter);
		if (current->Type == "Var") {
			continue;
		}
		else if (current->pChildren[0]->Type == "Var") {
			// print leave :)
			output_file << current->ID << ",NORMAL NODE," << current->scope.size() << endl;
			for (int i = 0; i < current->scope.size() - 1; i++) {
				output_file << current->scope[i] << ",";
			}
			output_file << current->scope.back() << endl;
			for (int i = 0; i < current->mmixture.mu.cols() - 1; i++) {
				output_file << current->mmixture.mu(i) << ",";
			}
			output_file << current->mmixture.W(current->mmixture.mu.cols() - 1) << endl;

			Eigen::MatrixXd Sigma(current->scope.size(), current->scope.size());
			Sigma = (1 / current->mmixture.v) * current->mmixture.W;
			
			for (int i = 0; i < Sigma.rows(); i++) {
				for (int j = 0; j < Sigma.cols(); j++) {
					output_file << Sigma(i, j);
					if (j != Sigma.cols() - 1) {
						output_file << ",";
					}
				}
				output_file << endl;
			}
		}
		else if (current->Type == "Sum") {
			// print SUM
			output_file << current->ID << ",SUM\n";
		}
		else {
			// print PRD
			output_file << current->ID << ",PRD\n";
		}
	}
	output_file << "##EDGES##\n";

	for (vector<spnNode*>::iterator iter = spnNetwork.begin(); iter<spnNetwork.end(); iter++) {
		spnNode* current = (*iter);
		if (current->Type == "Var") {
			continue;
		}
		else if (current->pChildren[0]->Type == "Var") {
			// print leave :)
			continue;
		}
		else if (current->Type == "Sum") {
			// print SUM
			for (int i = 0; i < current->children.size(); i++) {
				output_file << current->ID << "," << current->children[i] << "," << current->weights[i] << endl;
			}
		}
		else {
			// print PRD
			for (int i = 0; i < current->children.size(); i++) {
				output_file << current->ID << "," << current->children[i] << endl;
			}
		}
	}
	output_file.close();
}

void SPN::writeSPNMyFormat(std::string file_name) {
	ofstream output_file(file_name);
	for(vector<spnNode*>::iterator iter=spnNetwork.begin(); iter<spnNetwork.end(); iter++){
		output_file << "{\n";
		spnNode* current = (*iter);
		output_file << "\t" << current->ID << endl;
		output_file << "\t" << current->Type << endl;
		output_file << "\t" << current->children.size() <<endl;
		output_file << "\t[";
		for (int i=0; i < (int)(current->children.size()) -1; i++) {
			output_file << current->children[i] << ",";
		}
		if (current->children.size() > 0) {
			output_file << current->children.back();
		}
		output_file << "]\n";
		output_file << "\t" << current->parents.size() <<endl;
		output_file << "\t[";
		for (int i=0; i < (int)(current->parents.size())-1; i++) {
			output_file << current->parents[i] << ",";
		}
		if (current->parents.size() > 0) {
			output_file << current->parents.back();
		}
		output_file << "]\n";

		if (current->Type == "Var") {
			output_file << "\t[" << current->featureIdx << ", " << current->ValueIdx << "]\n";
		} else if (current->Type == "Sum") {
			output_file << "\t[";
			for (int i=0; i < (int)(current->weights.size())-1; i++) {
				output_file << current->weights[i] << ",";
			}
			if (current->weights.size() > 0) {
				output_file << current->weights.back();
			}
			output_file << "]\n";
			//output_file << "Scope: " << current->iscope[0] << " " << current->iscope[1] << " " << current->iscope[2] << " " << current->iscope[3] << endl;
		}
		output_file << "}\n";
	}
	output_file.close();
}

void SPN::writeDataCont(std::string filePath, std::vector<double*> data) {
	ofstream output_file(filePath);
	for (double *Instance : data) {
		for (int i = 0; i < numberOfVariables; i++) {
			output_file << Instance[i] << ",";
		}
		output_file << endl;
	}
	output_file.close();
}
void SPN::writeSPNMyFormatContinuous(std::string file_name) {
	ofstream output_file(file_name);
	for (vector<spnNode*>::iterator iter = spnNetwork.begin(); iter<spnNetwork.end(); iter++) {
		output_file << "{\n";
		spnNode* current = (*iter);
		output_file << "\t" << current->ID << endl;
		output_file << "\t" << current->Type << endl;
		output_file << "\t" << current->children.size() << endl;
		output_file << "\t[";
		for (int i = 0; i < (int)(current->children.size()) - 1; i++) {
			output_file << current->children[i] << ",";
		}
		if (current->children.size() > 0) {
			output_file << current->children.back();
		}
		output_file << "]\n";
		output_file << "\t" << current->parents.size() << endl;
		output_file << "\t[";
		for (int i = 0; i < (int)(current->parents.size()) - 1; i++) {
			output_file << current->parents[i] << ",";
		}
		if (current->parents.size() > 0) {
			output_file << current->parents.back();
		}
		output_file << "]\n";

		if (current->Type == "Var") {
			output_file << "\t[" << current->featureIdx << ", " << current->ValueIdx << "]\n";
		}
		else if (current->Type == "Sum") {
			output_file << "\t[";
			for (int i = 0; i < (int)(current->weights.size()) - 1; i++) {
				output_file << current->weights[i] << ",";
			}
			if (current->weights.size() > 0) {
				output_file << current->weights.back();
			}
			output_file << "]\n";
			output_file << "\t" << current->mixture_node << endl;
			if (current->mixture_node) {
				output_file << "\t" << current->gaussian_mixture.size() << endl;
				for (const mixture m : current->gaussian_mixture) {
					output_file << "\t" << m.NG[0] << "," << m.NG[1] << "," << m.NG[2] << "," << m.NG[3] <<endl;
				}
			}
		}
		output_file << "}\n";
	}
	output_file.close();
}
SPN::SPN(int numVariables, int numValues)
{
	numberOfVariables=numVariables;
	numberOfValues=numValues;
	numberOfMixtures = numValues;
	leaves.resize(numVariables);
	for (int i=0; i<numVariables; i++){
		leaves[i].resize(numValues);
	}
}

int SPN::readSPN(std::string filePath){
	ifstream spnFile;
	spnFile.open(filePath.c_str());
	if( spnFile.fail()){
		cout<<"Can't open spn file\n";
		return -1;
	}

	int i; //counter
	string input;
	char inChar;
	int inInt;
	double inDouble;

	int numberOfChildren;
	int numberOfParents;

	spnNode* newNode;

	while (!spnFile.eof()){
		spnFile>> input;
		if (input=="{"){
			newNode = new spnNode;


			spnFile>> inInt;
			if(inInt==0){ //root node
				root=inInt;
			}
			newNode->ID= inInt;

			spnFile>>input;
			newNode->Type=input;


			spnFile>>inInt; // number of children
			numberOfChildren=inInt;
			do{
				spnFile>> inChar;
			}while (inChar!= '[');
			for (i=0; i<numberOfChildren; i++){
				spnFile>>inInt;
				newNode->children.push_back(inInt);
				spnFile>>inChar;
			}
			while (inChar!= ']'){
				spnFile>> inChar;
			}


			spnFile>>inInt; // number of parents
			numberOfParents=inInt;
			do{
				spnFile>> inChar;
			}while (inChar!= '[');
			for (i=0; i<numberOfParents; i++){
				spnFile>>inInt;
				newNode->parents.push_back(inInt);
				spnFile>>inChar;
			}
			while (inChar!= ']'){
				spnFile>> inChar;
			}



			if (newNode->Type == "Sum"){

				do{
					spnFile>> inChar;
				}while (inChar!= '[');
				for (i=0; i<numberOfChildren; i++){
					spnFile>>inDouble;
					if (inDouble == 0) {
						cout<< "Weight is zero in the input tree?!\n";
						//assert();
					}
					newNode->weights.push_back(inDouble);
					newNode->log_weights.push_back(log(inDouble));
					newNode->log_inference.push_back(inDouble);
					spnFile>>inChar;
				}
				while (inChar!= ']'){
					spnFile>> inChar;
				}


			}else if (newNode->Type == "Product"){

			}else if (newNode->Type == "Var"){
				do{
					spnFile>> inChar;
				}while (inChar!= '[');

				spnFile>>inInt;
				newNode->featureIdx= inInt;

				spnFile>>inChar;

				spnFile>>inInt;
				newNode->ValueIdx= inInt;
				do{
					spnFile>> inChar;
				}while (inChar!= ']');
				leaves[newNode->featureIdx][newNode->ValueIdx]= newNode;
			}
			do{
				spnFile>> input;
			}while (input != "}");

		    spnNetwork.push_back(newNode);
		}
	}

	spnFile.close();

	for(vector<spnNode*>::iterator iter=spnNetwork.begin(); iter<spnNetwork.end(); iter++){
		spnNode* pointer= (*iter);
		for( vector<int>::iterator intIter=pointer->children.begin(); intIter< pointer->children.end(); intIter++){
			pointer->pChildren.push_back(spnNetwork[*intIter]);
		}
		for( vector<int>::iterator intIter=pointer->parents.begin(); intIter< pointer->parents.end(); intIter++){
			pointer->pParents.push_back(spnNetwork[*intIter]);
		}
	}
	cout<< "SPN Info: Number of nodes = "<< spnNetwork.size()<< endl;
	return 0;
}
int SPN::readSPNContinuousHan(std::string filePath) {
	ifstream spnFile;
	spnFile.open(filePath.c_str());
	if (spnFile.fail()) {
		std::cout << "Can't open spn file\n";
		return -1;
	}

	int i; //counter
	string input;
	char inChar;
	int inInt;
	double inDouble;

	int numberOfChildren;
	int numberOfParents;

	spnNode* newNode;
	root = 0;
	spnFile >> input; // ##NODES##
	if (input != "##NODES##") {
		std::cout << "Error: Wrong SPN file format ##NODES##\n";
	}


	while (!spnFile.eof()) {
		spnFile >> inInt;
		if (inInt != -1) {
			newNode = new spnNode;
			spnNetwork.push_back(newNode);

			newNode->ID = inInt;
			if (inInt == 0) {
				for (int j = 0; j < numberOfVariables; j++) {
					spnNode *newNode;
					newNode = new spnNode;
					newNode->ID = j + 1;
					newNode->Type = "Var";
					newNode->featureIdx = j;
					newNode->ValueIdx = 0;
					leaves[newNode->featureIdx][newNode->ValueIdx] = newNode;
					spnNetwork.push_back(newNode);
				}
			}

			newNode->mixture_node = false;
			spnFile >> inChar >> input;
			if (input == "SUM")
				newNode->Type = "Sum";
			else if (input == "PRD")
				newNode->Type = "Product";
			else if (input == "NORMAL") {
				newNode->Type = "Sum";
				spnFile >>  inChar >> inChar >> inChar >> inChar >> inChar >> inInt;
				int scope_size = inInt;
				vector<int> scope(inInt);
				spnFile >> inInt;
				scope[0] = inInt;
				for (int j = 1; j < scope_size; j++) {
					spnFile >> inChar >> inInt;
					scope[j] = inInt;
				}
				newNode->mixture_node = true;

				for (int j = 0; j < scope_size; j++) {
					newNode->scope.push_back(scope[j]);
					int current_var = newNode->scope[j];
					newNode->children.push_back(leaves[current_var][0]->ID);
					newNode->pChildren.push_back(leaves[current_var][0]);

					leaves[current_var][0]->parents.push_back(newNode->ID);
					leaves[current_var][0]->pParents.push_back(newNode);
				}
				spnFile >> inDouble;
				for (int j = 1; j < scope_size; j++)
					spnFile >> inChar >> inDouble;
				for (int j = 0; j < scope_size; j++) {
					spnFile >> inDouble;
					for (int k = 1; k < scope_size; k++) {
						spnFile >> inChar >> inDouble;
					}
				}
			}
		}
		else
			break;
	}
	spnFile >> input;
	if (input != "##EDGES##") {
		cout << "ERROR: Wrong format ##EDGES##\n";
	}
	int from, to;
	double weight;
	while (!spnFile.eof()) {
		spnFile >> from >> inChar >> to;
		if (spnNetwork[from]->Type == "Sum") {
			spnFile >> inChar >> weight;
			spnNetwork[from]->weights.push_back(weight);
			spnNetwork[from]->log_weights.push_back(log(weight));
		}
		spnNetwork[from]->children.push_back(to);
		spnNetwork[from]->pChildren.push_back(spnNetwork[to]);
		spnNetwork[from]->log_inference.push_back(0);

		spnNetwork[to]->parents.push_back(from);
		spnNetwork[to]->pParents.push_back(spnNetwork[from]);
	}

	spnFile.close();

/*	for (vector<spnNode*>::iterator iter = spnNetwork.begin(); iter<spnNetwork.end(); iter++) {
		spnNode* pointer = (*iter);
		for (vector<int>::iterator intIter = pointer->children.begin(); intIter< pointer->children.end(); intIter++) {
			pointer->pChildren.push_back(spnNetwork[*intIter]);
		}
		for (vector<int>::iterator intIter = pointer->parents.begin(); intIter< pointer->parents.end(); intIter++) {
			pointer->pParents.push_back(spnNetwork[*intIter]);
		}
	}*/
	cout << "SPN Info: Number of nodes = " << spnNetwork.size() << endl;
	return 0;
}
int SPN::readSPNContinuous(std::string filePath) {
	ifstream spnFile;
	spnFile.open(filePath.c_str());
	if (spnFile.fail()) {
		cout << "Can't open spn file\n";
		return -1;
	}

	int i; //counter
	string input;
	char inChar;
	int inInt;
	double inDouble;

	int numberOfChildren;
	int numberOfParents;

	spnNode* newNode;

	while (!spnFile.eof()) {
		spnFile >> input;
		if (input == "{") {
			newNode = new spnNode;


			spnFile >> inInt;
			if (inInt == 0) { //root node
				root = inInt;
			}
			newNode->ID = inInt;

			spnFile >> input;
			newNode->Type = input;


			spnFile >> inInt; // number of children
			numberOfChildren = inInt;
			do {
				spnFile >> inChar;
			} while (inChar != '[');
			for (i = 0; i<numberOfChildren; i++) {
				spnFile >> inInt;
				newNode->children.push_back(inInt);
				spnFile >> inChar;
			}
			while (inChar != ']') {
				spnFile >> inChar;
			}


			spnFile >> inInt; // number of parents
			numberOfParents = inInt;
			do {
				spnFile >> inChar;
			} while (inChar != '[');
			for (i = 0; i<numberOfParents; i++) {
				spnFile >> inInt;
				newNode->parents.push_back(inInt);
				spnFile >> inChar;
			}
			while (inChar != ']') {
				spnFile >> inChar;
			}



			if (newNode->Type == "Sum") {

				do {
					spnFile >> inChar;
				} while (inChar != '[');
				for (i = 0; i<numberOfChildren; i++) {
					spnFile >> inDouble;
					if (inDouble == 0) {
						cout << "Weight is zero in the input tree?!\n";
						//assert();
					}
					newNode->weights.push_back(inDouble);
					newNode->log_weights.push_back(log(inDouble));
					newNode->log_inference.push_back(inDouble);
					spnFile >> inChar;
				}
				while (inChar != ']') {
					spnFile >> inChar;
				}
				bool mixture_node;
				int number_of_gaussians;
				newNode->mixture_node = false;
				spnFile >> mixture_node;
				if (mixture_node) {
					spnFile >> number_of_gaussians;
					newNode->mixture_node = true;
					newNode->gaussian_mixture.resize(number_of_gaussians);
					newNode->updated_gaussian_mixture.resize(number_of_gaussians);
					newNode->updated_mixture_constants.resize(number_of_gaussians);
					for (int i = 0; i < number_of_gaussians; i++) {
						spnFile >> newNode->gaussian_mixture[i].NG[0] >> inChar >> newNode->gaussian_mixture[i].NG[1] >> inChar >> newNode->gaussian_mixture[i].NG[2] >> inChar >> newNode->gaussian_mixture[i].NG[3];
					}
				}

			}
			else if (newNode->Type == "Product") {

			}
			else if (newNode->Type == "Var") {
				do {
					spnFile >> inChar;
				} while (inChar != '[');

				spnFile >> inInt;
				newNode->featureIdx = inInt;

				spnFile >> inChar;

				spnFile >> inInt;
				newNode->ValueIdx = inInt;
				do {
					spnFile >> inChar;
				} while (inChar != ']');
				leaves[newNode->featureIdx][newNode->ValueIdx] = newNode;
			}
			do {
				spnFile >> input;
			} while (input != "}");

			spnNetwork.push_back(newNode);
		}
	}

	spnFile.close();

	for (vector<spnNode*>::iterator iter = spnNetwork.begin(); iter<spnNetwork.end(); iter++) {
		spnNode* pointer = (*iter);
		for (vector<int>::iterator intIter = pointer->children.begin(); intIter< pointer->children.end(); intIter++) {
			pointer->pChildren.push_back(spnNetwork[*intIter]);
		}
		for (vector<int>::iterator intIter = pointer->parents.begin(); intIter< pointer->parents.end(); intIter++) {
			pointer->pParents.push_back(spnNetwork[*intIter]);
		}
	}
	cout << "SPN Info: Number of nodes = " << spnNetwork.size() << endl;
	return 0;
}
int SPN::readtrainData(std::string filePath){
	ifstream dataFile;
	dataFile.open(filePath.c_str());
	if( dataFile.fail()){
		cout<<"Can't open spn file\n";
		return 1;
	}

	char inChar;
	int iFeature; //counters

	int* Instance;
	while(!dataFile.eof()){
		Instance=new int[numberOfVariables];
		dataFile>>Instance[0];
		if (dataFile.eof())
			break;
		for (iFeature=1; iFeature<numberOfVariables; iFeature++){
			dataFile>>inChar;
			dataFile>>Instance[iFeature];
		}
		trainData.push_back(Instance);
	}
	dataFile.close();
	cout<< "Train Data Info: Number of training examples = "<< trainData.size()<< endl;
	return 0;
}

int SPN::readtestData(std::string filePath){
	ifstream dataFile;
	dataFile.open(filePath.c_str());
	if( dataFile.fail()){
		cout<<"Can't open spn file\n";
		return 1;
	}

	char inChar;
	int iFeature; //counters

	int* Instance;
	while(!dataFile.eof()){
		Instance=new int[numberOfVariables];
		dataFile>>Instance[0];
		if (dataFile.eof())
			break;
		for (iFeature=1; iFeature<numberOfVariables; iFeature++){
			dataFile>>inChar;
			dataFile>>Instance[iFeature];
		}
		testData.push_back(Instance);
	}
	cout<< "Test Data Info: Number of test examples = "<< testData.size()<< endl;
	dataFile.close();
	return 0;
}


int SPN::readtrainDataCont(std::string filePath){
	ifstream dataFile;
	dataFile.open(filePath.c_str());
	if( dataFile.fail()){
		cout<<"Can't open spn file: " << filePath << "\n";
		return 1;
	}

	char inChar;
	int iFeature; //counters

	double* Instance;
	while(!dataFile.eof()){
		Instance=new double[numberOfVariables];
		dataFile>>Instance[0];
		if (dataFile.eof())
			break;
		for (iFeature=1; iFeature<numberOfVariables; iFeature++){
			dataFile>>inChar;
			dataFile>>Instance[iFeature];
		}
		trainDataCont.push_back(Instance);
	}
	dataFile.close();
	cout<< "Train Data Info: Number of training examples = "<< trainDataCont.size()<< endl;
	return 0;
}

int SPN::readtestDataCont(std::string filePath){
	ifstream dataFile;
	dataFile.open(filePath.c_str());
	if( dataFile.fail()){
		cout<<"Can't open testdatacont file\n";
		return 1;
	}

	char inChar;
	int iFeature; //counters

	double* Instance;
	while(!dataFile.eof()){
		Instance=new double[numberOfVariables];
		dataFile>>Instance[0];
		if (dataFile.eof())
			break;
		for (iFeature=1; iFeature<numberOfVariables; iFeature++){
			dataFile>>inChar;
			dataFile>>Instance[iFeature];
		}
		testDataCont.push_back(Instance);
	}
	cout<< "Test Data Info: Number of test examples = "<< testDataCont.size()<< endl;
	dataFile.close();
	return 0;
}





void SPN::normalizeWeights(double smoothing_factor) {
	for(vector<spnNode*>::iterator nodeIter=spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++){
		spnNode* current= (*nodeIter);
		if (current->Type == "Sum" && !current->mixture_node) {
			double sumWeights = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			if (sumWeights == 0 ) {
				cout << " Weights: ";
				for (int i = 0; i < current->weights.size(); i++) {
					cout << current->weights[i] << " ";
				}
				cout << endl;
				exit (EXIT_FAILURE);
			}
			vector<double>::iterator log_weights = current->log_weights.begin();
			for (vector<double>::iterator node = current->weights.begin(); node < current->weights.end(); node++, log_weights++) {
				*node = ((*node) + smoothing_factor/(double)current->weights.size()) / (sumWeights + smoothing_factor);
				*log_weights = log(*node);
			}
		}
	}
}

void SPN::randomizeWeights() {
	for(vector<spnNode*>::iterator nodeIter=spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++){
		spnNode* current= (*nodeIter);
		if (current->Type == "Sum") {
			for (int i=0; i < current->weights.size(); i++) {
				current->weights[i] = rand() + 1;
				current->log_weights[i] = log(current->weights[i]);
			}
		}
	}
}

void SPN::randomizeWeightsContinuous() {
	for (vector<spnNode*>::iterator nodeIter = spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++) {
		spnNode* current = (*nodeIter);
		if (current->Type == "Sum") {
			for (int i = 0; i < current->weights.size(); i++) {
				current->weights[i] = rand() % 5 + 1;
				current->log_weights[i] = log(current->weights[i]);
			}
			if (current->mixture_node) {
				for (int i = 0; i < current->gaussian_mixture.size(); i++) {
					current->gaussian_mixture[i].NG[0] = ((rand()) % 20 - rand() % 20) / 20.0;
					current->gaussian_mixture[i].NG[1] = 0.1;
					current->gaussian_mixture[i].NG[2] = (rand()) % 10  + 2;
					current->gaussian_mixture[i].NG[3] = 2;
				}
			}
		}
	}
}
void SPN::randomizeWeightsContinuousMultivariate() {
	int params = 0;
	for (vector<spnNode*>::iterator nodeIter = spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++) {
		spnNode* current = (*nodeIter);
		if (current->Type == "Sum") {
			for (int i = 0; i < current->weights.size(); i++) {
				current->weights[i] = rand() % 5 + 1;
				current->log_weights[i] = log(current->weights[i]);
				params++;
			}
			if (current->mixture_node) {
				current->mmixture.v = (rand() % 5) + 2 + current->scope.size();
				current->mmixture.k = (rand() % 5) + 2;
				current->mmixture.mu.resize(1, current->scope.size());
				current->mmixture.mu.setRandom(1, current->scope.size());
				current->mmixture.W.resize(current->scope.size(), current->scope.size());	
				current->mmixture.W.setRandom(current->scope.size(), current->scope.size());
				//current->mmixture.W = current->mmixture.W.cwiseAbs();
				current->mmixture.W = Eigen::MatrixXd(current->mmixture.W.triangularView<Eigen::Lower>());
				//cout << "1" << endl;
				
				current->mmixture.W = current->mmixture.W * current->mmixture.W.transpose();
				current->mmixture.W += Eigen::MatrixXd::Identity(current->scope.size(), current->scope.size());
				//current->mmixture.W = pow(1 / current->mmixture.W.determinant(), 1.0 / (double)current->scope.size()) *current->mmixture.W;

				//cout << current->mmixture.W;
				//current->mmixture.W = current->scope.size() * current->mmixture.W;
				//cout << endl <<  current->mmixture.W.determinant() << endl;
				//cout << "2" << endl;
				//cout << current->mmixture.W;
				//current->mmixture.W =  (1/ current->mmixture.W.determinant()) * current->mmixture.W;
				//cout << endl << current->mmixture.W << endl;
				//cout << current->mmixture.W.determinant() << endl;
				params = params + 2 + current->scope.size() + pow(current->scope.size(), 2);
			}
		}
	}
	cout << "Number of Parameters = " << params << endl;
}

void SPN::printMixtures() {
	for (vector<spnNode*>::iterator nodeIter = spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++) {
		spnNode* current = (*nodeIter);
		if (current->Type == "Sum") {
			if (current->mixture_node) {
				cout << "Mu: \n" << current->mmixture.mu << endl;
				cout << "Sigma: \n" << 1/current->mmixture.v * current->mmixture.W << endl;
			}
			else {
				cout << "Weights: \n";
				for (const double w : current->weights) {
					cout << w << " ";
				}
				cout << endl;
			}
		}
	}
}
double SPN::doInferenceOnInstance(int* Instance){
	for (int i=0; i<numberOfVariables; i++) {
		for (int j=0; j<numberOfValues; j++) {
			leaves[i][j]->inferenceValue = 0;
			leaves[i][j]->log_inferenceValue = -1000;
			if (j == Instance[i]) {
				leaves[i][j]->inferenceValue = 1;
				leaves[i][j]->log_inferenceValue = log(1.0);
			}
		}
	}
	for(vector<spnNode*>::reverse_iterator nodeIter=spnNetwork.rbegin(); nodeIter<spnNetwork.rend(); nodeIter++){
		spnNode* current= (*nodeIter);
		if(current->Type == "Sum"){
			double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			vector<double>::iterator currentWeight= current->log_weights.begin();
			vector<double>::iterator current_log_inference = current->log_inference.begin();
			double maximum;
			for (vector<spnNode*>::iterator currentChild= current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++, currentWeight++, current_log_inference++){
				*current_log_inference = (*currentChild)->log_inferenceValue + *currentWeight;
				if (currentChild == current->pChildren.begin())
				{
					maximum = *current_log_inference;
				} else if (maximum < *current_log_inference) {
					maximum = *current_log_inference;
				}
			}
			transform( current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(),bind2nd(std::plus<double>(), 0.0 - maximum));
			double sum=0;
			for(vector<double>::iterator w = current->log_inference.begin(); w < current->log_inference.end(); w++) {
				sum += exp(*w);
			}
			current->log_inferenceValue= log(sum) - log(weightSum) + maximum;
			current->inferenceValue = exp(current->log_inferenceValue);
		}else if (current->Type == "Product"){
			double product=0;
			for (vector<spnNode*>::iterator currentChild= current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++){
				product+= (*currentChild)->log_inferenceValue;
			}
			current->log_inferenceValue = product;
			current->inferenceValue = exp(product);
		}
	}
	return spnNetwork[root]->log_inferenceValue;
}

double SPN::doInferenceOnInstanceContinuous(double* Instance) {
	for (int i = 0; i<numberOfVariables; i++) {
		for (int j = 0; j<numberOfValues; j++) {
			leaves[i][j]->inferenceValue = Instance[i];
		}
	}
	for (vector<spnNode*>::reverse_iterator nodeIter = spnNetwork.rbegin(); nodeIter<spnNetwork.rend(); nodeIter++) {
		spnNode* current = (*nodeIter);
		if (current->Type == "Sum" && current->mixture_node) {
			double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			vector<double>::iterator currentWeight = current->log_weights.begin();
			vector<double>::iterator current_log_inference = current->log_inference.begin();
			double maximum = 0;
			double observation = current->pChildren[0]->inferenceValue;
			for (vector<mixture>::iterator current_mixture = current->gaussian_mixture.begin(); current_mixture < current->gaussian_mixture.end(); current_mixture++, currentWeight++, current_log_inference++) {

				double log_inference;
				current_mixture->Normal[0] = current_mixture->NG[0];
				if (current_mixture->NG[2] == 0) {
					current_mixture->NG[2] += 0.0000001;
					cout << "Warning: beta is zero\n";
				}
				if (current_mixture->NG[3] == 0)
				{
					current_mixture->NG[3] += 0.0000001;
					cout << "Warning: sigma is zero\n";
				}

				current_mixture->Normal[1] = current_mixture->NG[3] / current_mixture->NG[2] ;
				log_inference = -0.5*log(current_mixture->Normal[1]) - 0.5 * log(2 * M_PI) - 0.5 * pow((observation - current_mixture->Normal[0]),2) / current_mixture->Normal[1];
				*current_log_inference = log_inference + *currentWeight;
				if (current_mixture == current->gaussian_mixture.begin())
				{
					maximum = *current_log_inference;
				}
				else if (maximum < *current_log_inference) {
					maximum = *current_log_inference;
				}
			}
			transform(current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(), bind2nd(std::plus<double>(), 0.0 - maximum));
			double sum = 0;
			for (vector<double>::iterator w = current->log_inference.begin(); w < current->log_inference.end(); w++) {
				sum += exp(*w);
			}
			current->log_inferenceValue = log(sum) - log(weightSum) + maximum;
			current->inferenceValue = exp(current->log_inferenceValue);
		} else if (current->Type == "Sum") {
			double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			vector<double>::iterator currentWeight = current->log_weights.begin();
			vector<double>::iterator current_log_inference = current->log_inference.begin();
			double maximum;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++, currentWeight++, current_log_inference++) {
				*current_log_inference = (*currentChild)->log_inferenceValue + *currentWeight;
				if (currentChild == current->pChildren.begin())
				{
					maximum = *current_log_inference;
				}
				else if (maximum < *current_log_inference) {
					maximum = *current_log_inference;
				}
			}
			transform(current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(), bind2nd(std::plus<double>(), 0.0 - maximum));
			double sum = 0;
			for (vector<double>::iterator w = current->log_inference.begin(); w < current->log_inference.end(); w++) {
				sum += exp(*w);
			}
			current->log_inferenceValue = log(sum) - log(weightSum) + maximum;
			current->inferenceValue = exp(current->log_inferenceValue);
		}
		else if (current->Type == "Product") {
			double product = 0;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++) {
				product += (*currentChild)->log_inferenceValue;
			}
			current->log_inferenceValue = product;
			current->inferenceValue = exp(product);
		}
	}
	return spnNetwork[root]->log_inferenceValue;
}

double SPN::doInferenceOnInstanceContinuousMultivariate(double* Instance) {
	for (int i = 0; i<numberOfVariables; i++) {
		for (int j = 0; j<numberOfValues; j++) {
			leaves[i][j]->inferenceValue = Instance[i];
		}
	}
	for (vector<spnNode*>::reverse_iterator nodeIter = spnNetwork.rbegin(); nodeIter<spnNetwork.rend(); nodeIter++) {
		spnNode* current = (*nodeIter);
		if (current->Type == "Sum" && current->mixture_node) {
			//double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			//vector<double>::iterator currentWeight = current->log_weights.begin();
			//vector<double>::iterator current_log_inference = current->log_inference.begin();
			//double maximum = 0;
			//double observation = current->pChildren[0]->inferenceValue;

			int d = current->scope.size();
			Eigen::MatrixXd X(1,d);
			Eigen::MatrixXd Mu(1,d);
			Mu = current->mmixture.mu;
			Eigen::MatrixXd Sigma(d,d); // Sigma needs to be updated

			Sigma = (1/current->mmixture.v) * current->mmixture.W;

			for (int i = 0; i < current->pChildren.size(); i++) {
				X(0, i) = current->pChildren[i]->inferenceValue;
			}

			
			Eigen::MatrixXd XMu(1, d);
			Eigen::MatrixXd result(d, d);


			XMu = X - Mu;
			result = (XMu) * Sigma.inverse() * (XMu.transpose());
			//cout << "Sigma: " << Sigma.determinant() << endl;
			//cout << "result: " << result.determinant() << endl;
			
			double log_inference;
		    log_inference = -0.5*log(Sigma.determinant()) - 0.5 * d * log(2 * M_PI) - 0.5 * result.determinant();

			current->log_inferenceValue = log_inference;
			current->inferenceValue = exp(current->log_inferenceValue);
		}
		else if (current->Type == "Sum") {
			double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			vector<double>::iterator currentWeight = current->log_weights.begin();
			vector<double>::iterator current_log_inference = current->log_inference.begin();
			double maximum;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++, currentWeight++, current_log_inference++) {
				*current_log_inference = (*currentChild)->log_inferenceValue + *currentWeight;
				if (currentChild == current->pChildren.begin())
				{
					maximum = *current_log_inference;
				}
				else if (maximum < *current_log_inference) {
					maximum = *current_log_inference;
				}
			}
			transform(current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(), bind2nd(std::plus<double>(), 0.0 - maximum));
			double sum = 0;
			for (vector<double>::iterator w = current->log_inference.begin(); w < current->log_inference.end(); w++) {
				sum += exp(*w);
			}
			current->log_inferenceValue = log(sum) - log(weightSum) + maximum;
			current->inferenceValue = exp(current->log_inferenceValue);
		}
		else if (current->Type == "Product") {
			double product = 0;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++) {
				product += (*currentChild)->log_inferenceValue;
			}
			current->log_inferenceValue = product;
			current->inferenceValue = exp(product);
		}
	}
	return spnNetwork[root]->log_inferenceValue;
}




double SPN::doInference(std::vector<int*> &data, string file_path) {
	double LL=0;
	ofstream output_file(file_path);
	for (int i=0; i<data.size(); i++){
		double value=doInferenceOnInstance(data[i]);
		output_file << "Test sample #: " << i << " log likelihood: " << value << endl;
		LL+= doInferenceOnInstance(data[i]);
	}
	output_file.close();
	cout << "Average Log Likelihood on the dataset is : " << LL/data.size() << endl;
	return LL/data.size();
}

double SPN::doInferenceContinuous(std::vector<double*> &data, string file_path) {
	double LL = 0;
	ofstream output_file(file_path);
	for (int i = 0; i<data.size(); i++) {
		double value = doInferenceOnInstanceContinuous(data[i]);
		//output_file << "Test sample #: " << i << " log likelihood: " << value << endl;
		output_file << value << endl;
		LL += doInferenceOnInstanceContinuous(data[i]);
	}
	output_file.close();
	cout << "Average Log Likelihood on the dataset is : " << LL / data.size() << endl;
	return LL / data.size();
}

double SPN::doInferenceContinuousMultivariate(std::vector<double*> &data, string file_path) {
	double LL = 0;
	ofstream output_file(file_path);
	for (int i = 0; i<data.size(); i++) {
		double value = doInferenceOnInstanceContinuousMultivariate(data[i]);
		//output_file << "Test sample #: " << i << " log likelihood: " << value << endl;
		output_file << value << endl;
		LL += doInferenceOnInstanceContinuousMultivariate(data[i]);
	}
	output_file.close();
	cout << "Average Log Likelihood on the dataset is : " << LL / data.size() << endl;
	return LL / data.size();
}

double SPN::doInferenceOnUCI(std::string ifile) {
	double LL=0;
	ifstream uci_file(ifile);
	int number_of_documents, number_of_words, total_words, temp;
	uci_file >> number_of_documents >> number_of_words >> total_words;
	int *vec = new int[number_of_words];
	
	double current_document;
	int word;
	uci_file >> current_document;
	for (double i=0; i<number_of_documents; i++) {
		if ((int) i% 1000 == 0) {
			cout << "Processing instance: " << i << " of:" << number_of_documents<<endl;
		}
		for (int j=0; j < number_of_words; j++) {
			vec[j] = 0;
		}
		while (current_document == i + 1 && !uci_file.eof()) {
			uci_file >> word;
			vec[ word - 1] = 1;
			uci_file >> temp;
			uci_file >> current_document;
		}
		LL+= doInferenceOnInstance(vec);
	}
	uci_file.close();
	cout << "Average Log Likelihood on the dataset is : " << LL/number_of_documents << endl;
	return LL/number_of_documents;
}
void SPN::doUpwardPass(int *Instance) {
	for (int i=0; i<numberOfVariables; i++) {
		for (int j=0; j<numberOfValues; j++) {
			leaves[i][j]->inferenceValue = 0;
			leaves[i][j]->log_inferenceValue = -1000;
			if (j == Instance[i]) {
				leaves[i][j]->inferenceValue = 1;
				leaves[i][j]->log_inferenceValue = log(1.0);
			}
		}
	}
	for(vector<spnNode*>::reverse_iterator nodeIter=spnNetwork.rbegin(); nodeIter<spnNetwork.rend(); nodeIter++){
		spnNode* current= (*nodeIter);

		if(current->Type == "Sum"){
			double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			vector<double>::iterator currentWeight= current->log_weights.begin();
			vector<double>::iterator current_log_inference = current->log_inference.begin();
			double maximum;
			for (vector<spnNode*>::iterator currentChild= current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++, currentWeight++, current_log_inference++){
				*current_log_inference = (*currentChild)->log_inferenceValue + *currentWeight;
				if (currentChild == current->pChildren.begin())
				{
					maximum = *current_log_inference;
				} else if (maximum < *current_log_inference) {
					maximum = *current_log_inference;
				}
			}
			transform( current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(),bind2nd(std::plus<double>(), 0.0 - maximum));
			double sum=0;
			for(vector<double>::iterator w = current->log_inference.begin(); w < current->log_inference.end(); w++) {
				sum += exp(*w);
			}
			if (isnan(sum)) {
				cout << "sum inference is nan: " << endl << "log inference of children: ";
				for (int l=0; l < current->log_inference.size(); l++) {
					cout << current->log_inference[l] << " ";
				}

				cout <<endl<< "Weights: ";
				for (int l=0; l < current->weights.size(); l++) {
					cout << current->weights[l] << " ";
				}

				cout <<endl<< "id: ";
				for (int l=0; l < current->weights.size(); l++) {
					cout << current->children[l] << " ";
				}
				cout << endl;
				exit (EXIT_FAILURE);

			}
			current->log_inferenceValue= log(sum) - log(weightSum) + maximum;
			current->inferenceValue = exp(current->log_inferenceValue);
		}else if (current->Type == "Product"){
			double product=0;
			for (vector<spnNode*>::iterator currentChild= current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++){
				product += (*currentChild)->log_inferenceValue;
			}
			current->log_inferenceValue = product;
			current->inferenceValue = exp(product);
		}
	}
}

void SPN::doUpwardPassContinuous(double *Instance) {
	for (int i = 0; i<numberOfVariables; i++) {
		for (int j = 0; j<numberOfValues; j++) {
			leaves[i][j]->inferenceValue = Instance[i];
		}
	}
	for (vector<spnNode*>::reverse_iterator nodeIter = spnNetwork.rbegin(); nodeIter<spnNetwork.rend(); nodeIter++) {
		spnNode* current = (*nodeIter);

		if (current->Type == "Sum" && current->mixture_node) {
			double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);

			double maximum=0;
			double observation = current->pChildren[0]->inferenceValue;
			for (int i = 0; i < current->gaussian_mixture.size(); i++) {
				double updated_value;
				// mu
				updated_value = (current->gaussian_mixture[i].NG[1] * current->gaussian_mixture[i].NG[0] + observation) / (current->gaussian_mixture[i].NG[1] + 1);
				if (isnan(updated_value)) {
					cout << "Error: updated mu is nan\n";
					exit(EXIT_FAILURE);
				}
				current->updated_gaussian_mixture[i].NG[0] = updated_value;

				// k
				current->updated_gaussian_mixture[i].NG[1] = current->gaussian_mixture[i].NG[1] + 1;
				// alpha
				current->updated_gaussian_mixture[i].NG[2] = current->gaussian_mixture[i].NG[2] + 0.5;

				// beta
				current->updated_gaussian_mixture[i].NG[3] = current->gaussian_mixture[i].NG[3] + current->gaussian_mixture[i].NG[1] * pow(observation - current->gaussian_mixture[i].NG[0], 2) / (2* (current->gaussian_mixture[i].NG[1] + 1));
				//double z = (current->gaussian_mixture[i].NG[1] * pow(current->gaussian_mixture[i].NG[0], 2) + pow(observation, 2)) / (current->gaussian_mixture[i].NG[1] + 1);
				//current->updated_gaussian_mixture[i].NG[3] = current->gaussian_mixture[i].NG[3] + (current->gaussian_mixture[i].NG[1]+1) * ( z - pow(current->updated_gaussian_mixture[i].NG[0], 2)) / 2;


				current->updated_mixture_constants[i] = 0.5 * log(current->gaussian_mixture[i].NG[1]) - 0.5 *log(current->updated_gaussian_mixture[i].NG[1]) 
					+ lgamma(current->updated_gaussian_mixture[i].NG[2]) - lgamma(current->gaussian_mixture[i].NG[2]) 
					+ current->gaussian_mixture[i].NG[2] * log(current->gaussian_mixture[i].NG[3]) - current->updated_gaussian_mixture[i].NG[2] * log(current->updated_gaussian_mixture[i].NG[3]);
				//if (current->updated_mixture_constants[i] > 0) {
				//	cout << current->updated_mixture_constants[i] << endl;
				//}
			}

			vector<double>::iterator currentWeight = current->log_weights.begin();
			vector<double>::iterator current_log_inference = current->log_inference.begin();
			vector<double>::iterator current_updated_mixture_constant = current->updated_mixture_constants.begin();
			for (vector<double>::iterator current_updated_mixture_constant = current->updated_mixture_constants.begin(); current_updated_mixture_constant< current->updated_mixture_constants.end(); currentWeight++, current_log_inference++, current_updated_mixture_constant++) {
				*current_log_inference = *current_updated_mixture_constant + *currentWeight ;
				if (current_updated_mixture_constant == current->updated_mixture_constants.begin())
				{
					maximum = *current_log_inference;
				}
				else if (maximum < *current_log_inference) {
					maximum = *current_log_inference;
				}
			}
			transform(current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(), bind2nd(std::plus<double>(), 0.0 - maximum));
			double sum = 0;
			for (vector<double>::iterator w = current->log_inference.begin(); w < current->log_inference.end(); w++) {
				sum += exp(*w);
			}
			if (isnan(sum)) {
				cout << "sum inference is nan: " << endl << "log inference of children: ";
				for (int l = 0; l < current->log_inference.size(); l++) {
					cout << current->log_inference[l] << " ";
				}

				cout << endl << "Weights: ";
				for (int l = 0; l < current->weights.size(); l++) {
					cout << current->weights[l] << " ";
				}

				cout << endl << "id: ";
				for (int l = 0; l < current->weights.size(); l++) {
					cout << current->children[l] << " ";
				}
				cout << endl;
				exit(EXIT_FAILURE);

			}
			transform(current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(), bind2nd(std::plus<double>(), 0.0 + maximum));
			current->log_inferenceValue = log(sum) - log(weightSum) + maximum;
			current->inferenceValue = exp(current->log_inferenceValue);
		} else if (current->Type == "Sum") {
			double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			vector<double>::iterator currentWeight = current->log_weights.begin();
			vector<double>::iterator current_log_inference = current->log_inference.begin();
			double maximum;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++, currentWeight++, current_log_inference++) {
				*current_log_inference = (*currentChild)->log_inferenceValue + *currentWeight;
				if (currentChild == current->pChildren.begin())
				{
					maximum = *current_log_inference;
				}
				else if (maximum < *current_log_inference) {
					maximum = *current_log_inference;
				}
			}
			transform(current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(), bind2nd(std::plus<double>(), 0.0 - maximum));
			double sum = 0;
			for (vector<double>::iterator w = current->log_inference.begin(); w < current->log_inference.end(); w++) {
				sum += exp(*w);
			}
			if (isnan(sum)) {
				cout << "sum inference is nan: " << endl << "log inference of children: ";
				for (int l = 0; l < current->log_inference.size(); l++) {
					cout << current->log_inference[l] << " ";
				}

				cout << endl << "Weights: ";
				for (int l = 0; l < current->weights.size(); l++) {
					cout << current->weights[l] << " ";
				}

				cout << endl << "id: ";
				for (int l = 0; l < current->weights.size(); l++) {
					cout << current->children[l] << " ";
				}
				cout << endl;
				exit(EXIT_FAILURE);

			}
			current->log_inferenceValue = log(sum) - log(weightSum) + maximum;
			current->inferenceValue = exp(current->log_inferenceValue);
		}
		else if (current->Type == "Product") {
			double product = 0;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++) {
				product += (*currentChild)->log_inferenceValue;
			}
			current->log_inferenceValue = product;
			current->inferenceValue = exp(product);
		}
	}
}


void SPN::doUpwardPassContinuous1(double *Instance) {
	for (int i = 0; i<numberOfVariables; i++) {
		for (int j = 0; j<numberOfValues; j++) {
			leaves[i][j]->inferenceValue = Instance[i];
		}
	}
	for (vector<spnNode*>::reverse_iterator nodeIter = spnNetwork.rbegin(); nodeIter<spnNetwork.rend(); nodeIter++) {
		spnNode* current = (*nodeIter);

		if (current->Type == "Sum" && current->mixture_node) {
			double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			vector<double>::iterator currentWeight = current->log_weights.begin();
			vector<double>::iterator current_log_inference = current->log_inference.begin();
			double maximum = 0;
			double observation = current->pChildren[0]->inferenceValue;
			for (vector<mixture>::iterator current_mixture = current->gaussian_mixture.begin(); current_mixture < current->gaussian_mixture.end(); current_mixture++, currentWeight++, current_log_inference++) {

				double log_inference;
				current_mixture->Normal[0] = current_mixture->NG[0];
				if (current_mixture->NG[2] == 0) {
					current_mixture->NG[2] += 0.0000001;
					cout << "Warning: beta is zero\n";
				}
				if (current_mixture->NG[3] == 0)
				{
					current_mixture->NG[3] += 0.0000001;
					cout << "Warning: sigma is zero\n";
				}

				current_mixture->Normal[1] = current_mixture->NG[3] / current_mixture->NG[2];
				log_inference = -0.5*log(current_mixture->Normal[1]) - 0.5 * log(2 * M_PI) - 0.5 * pow((observation - current_mixture->Normal[0]), 2) / current_mixture->Normal[1];
				*current_log_inference = log_inference + *currentWeight;
				if (current_mixture == current->gaussian_mixture.begin())
				{
					maximum = *current_log_inference;
				}
				else if (maximum < *current_log_inference) {
					maximum = *current_log_inference;
				}
			}
			transform(current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(), bind2nd(std::plus<double>(), 0.0 - maximum));
			double sum = 0;
			for (vector<double>::iterator w = current->log_inference.begin(); w < current->log_inference.end(); w++) {
				sum += exp(*w);
			}
			current->log_inferenceValue = log(sum) - log(weightSum) + maximum;
			current->inferenceValue = exp(current->log_inferenceValue);
		}
		else if (current->Type == "Sum") {
			double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			vector<double>::iterator currentWeight = current->log_weights.begin();
			vector<double>::iterator current_log_inference = current->log_inference.begin();
			double maximum;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++, currentWeight++, current_log_inference++) {
				*current_log_inference = (*currentChild)->log_inferenceValue + *currentWeight;
				if (currentChild == current->pChildren.begin())
				{
					maximum = *current_log_inference;
				}
				else if (maximum < *current_log_inference) {
					maximum = *current_log_inference;
				}
			}
			transform(current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(), bind2nd(std::plus<double>(), 0.0 - maximum));
			double sum = 0;
			for (vector<double>::iterator w = current->log_inference.begin(); w < current->log_inference.end(); w++) {
				sum += exp(*w);
			}
			if (isnan(sum)) {
				cout << "sum inference is nan: " << endl << "log inference of children: ";
				for (int l = 0; l < current->log_inference.size(); l++) {
					cout << current->log_inference[l] << " ";
				}

				cout << endl << "Weights: ";
				for (int l = 0; l < current->weights.size(); l++) {
					cout << current->weights[l] << " ";
				}

				cout << endl << "id: ";
				for (int l = 0; l < current->weights.size(); l++) {
					cout << current->children[l] << " ";
				}
				cout << endl;
				exit(EXIT_FAILURE);

			}
			current->log_inferenceValue = log(sum) - log(weightSum) + maximum;
			current->inferenceValue = exp(current->log_inferenceValue);
		}
		else if (current->Type == "Product") {
			double product = 0;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++) {
				product += (*currentChild)->log_inferenceValue;
			}
			current->log_inferenceValue = product;
			current->inferenceValue = exp(product);
		}
	}
}

void SPN::doUpwardPassContinuousMultivariate(double *Instance) {
	for (int i = 0; i<numberOfVariables; i++) {
		for (int j = 0; j<numberOfValues; j++) {
			leaves[i][j]->inferenceValue = Instance[i];
		}
	}
	for (vector<spnNode*>::reverse_iterator nodeIter = spnNetwork.rbegin(); nodeIter<spnNetwork.rend(); nodeIter++) {
		spnNode* current = (*nodeIter);

		if (current->Type == "Sum" && current->mixture_node) {
			//double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			//vector<double>::iterator currentWeight = current->log_weights.begin();
			//vector<double>::iterator current_log_inference = current->log_inference.begin();
			//double maximum = 0;
			//double observation = current->pChildren[0]->inferenceValue;
			int d = current->scope.size();
			Eigen::MatrixXd X(1, d);
			for (int i = 0; i < current->pChildren.size(); i++) {
				X(0, i) = current->pChildren[i]->inferenceValue;
			}
			
			current->updated_mmixture.k = current->mmixture.k + 1;
			current->updated_mmixture.v = current->mmixture.v + 1;
			current->updated_mmixture.mu = 1 / (current->updated_mmixture.k) * (X + current->mmixture.k * current->mmixture.mu);
			Eigen::MatrixXd XMu = X - current->mmixture.mu;
			current->updated_mmixture.W = current->mmixture.W + (current->mmixture.k) / (current->updated_mmixture.k) * ((XMu.transpose())*(XMu));
			double B, C, D; // constants
			B = ((double)d / 2.0) * log(current->mmixture.k / current->updated_mmixture.k);
			C = current->mmixture.v * log(current->mmixture.W.determinant()) - current->updated_mmixture.v * log(current->updated_mmixture.W.determinant());
			if (current->updated_mmixture.v < d) {
				D = lgamma(0.5 * current->updated_mmixture.v) - lgamma(1);
			}
			else {
				D = lgamma(0.5 * current->updated_mmixture.v) - lgamma(0.5*(current->updated_mmixture.v-d));
			}

			current->mmixture_constant = log(0.5*current->mmixture.v) + B + C + D;

			//double log_inference;
			//log_inference = -0.5*log(Sigma.determinant()) - 0.5 * d * log(2 * M_PI) - 0.5 * result.determinant();

			current->log_inferenceValue = current->mmixture_constant;
			current->inferenceValue = exp(current->log_inferenceValue);
		}
		else if (current->Type == "Sum") {
			double weightSum = std::accumulate(current->weights.begin(), current->weights.end(), 0.0);
			vector<double>::iterator currentWeight = current->log_weights.begin();
			vector<double>::iterator current_log_inference = current->log_inference.begin();
			double maximum;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++, currentWeight++, current_log_inference++) {
				*current_log_inference = (*currentChild)->log_inferenceValue + *currentWeight;
				if (currentChild == current->pChildren.begin())
				{
					maximum = *current_log_inference;
				}
				else if (maximum < *current_log_inference) {
					maximum = *current_log_inference;
				}
			}
			transform(current->log_inference.begin(), current->log_inference.end(), current->log_inference.begin(), bind2nd(std::plus<double>(), 0.0 - maximum));
			double sum = 0;
			for (vector<double>::iterator w = current->log_inference.begin(); w < current->log_inference.end(); w++) {
				sum += exp(*w);
			}
			if (isnan(sum)) {
				cout << "sum inference is nan: " << endl << "log inference of children: ";
				for (int l = 0; l < current->log_inference.size(); l++) {
					cout << current->log_inference[l] << " ";
				}

				cout << endl << "Weights: ";
				for (int l = 0; l < current->weights.size(); l++) {
					cout << current->weights[l] << " ";
				}

				cout << endl << "id: ";
				for (int l = 0; l < current->weights.size(); l++) {
					cout << current->children[l] << " ";
				}
				cout << endl;
				exit(EXIT_FAILURE);

			}
			current->log_inferenceValue = log(sum) - log(weightSum) + maximum;
			current->inferenceValue = exp(current->log_inferenceValue);
		}
		else if (current->Type == "Product") {
			double product = 0;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++) {
				product += (*currentChild)->log_inferenceValue;
			}
			current->log_inferenceValue = product;
			current->inferenceValue = exp(product);
		}
	}
}
void SPN::doDownwardPass(int *Instance) {
	/*for(vector<spnNode*>::iterator nodeIter=spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++){
		spnNode* current= (*nodeIter);

		if(current->Type == "Sum" && current->ID != root){
			spnNode* prod = current->pParents[0];
			spnNode* parentSum = prod->pParents[0];
			double C1=0;
			double C2=0;
			double weightSum = std::accumulate(parentSum->weights.begin(), parentSum->weights.end(), 0.0);
			vector<double>::iterator currentWeight= parentSum->weights.begin();
			for (vector<spnNode*>::iterator currentChild= parentSum->pChildren.begin(); currentChild< parentSum->pChildren.end(); currentChild++, currentWeight++){
				if ( (*currentChild)!= prod )
					C1+= (*currentChild)->inferenceValue * (*currentWeight) / weightSum;
				else
					C2= (*currentWeight) / weightSum;
			}
			double constProd=1;
			for (vector<spnNode*>::iterator currentChild= prod->pChildren.begin(); currentChild< prod->pChildren.end(); currentChild++){
				if ( (*currentChild)!= current )
					constProd *= (*currentChild)->inferenceValue;
			}
			current->learning_vector[2]= C2*constProd*parentSum->learning_vector[2];
			current->learning_vector[3]= C2*constProd*parentSum->learning_vector[3];
			current->learning_vector[4]= C2*constProd*parentSum->learning_vector[4];
			current->learning_vector[5]= parentSum->learning_vector[5] + C1*parentSum->learning_vector[4];
			current->learning_vector[0]= current->learning_vector[2]+current->learning_vector[3];
			current->learning_vector[1]= current->learning_vector[5];
		}
	}*/

	for (vector<spnNode*>::iterator nodeIter = spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++) {
		spnNode* current = (*nodeIter);

		if (current->Type == "Sum" && current->ID != root) {
			spnNode* prod = current->pParents[0];
			spnNode* parentSum = prod->pParents[0];
			double C1 = 0, log_C1 = 0;
			double C2 = 0;
			double weightSum = std::accumulate(parentSum->weights.begin(), parentSum->weights.end(), 0.0);


			double maximum = parentSum->pChildren[0]->log_inferenceValue;
			for (const spnNode* s : parentSum->pChildren) {
				if (s->log_inferenceValue > maximum) {
					maximum = s->log_inferenceValue;
				}
			}

			vector<double>::iterator currentWeight = parentSum->weights.begin();
			for (vector<spnNode*>::iterator currentChild = parentSum->pChildren.begin(); currentChild< parentSum->pChildren.end(); currentChild++, currentWeight++) {
				if ((*currentChild) != prod)
					C1 += exp((*currentChild)->log_inferenceValue + log((*currentWeight) / weightSum) - maximum);
				else
					C2 = (*currentWeight) / weightSum;
			}
			log_C1 = log(C1) + maximum;
			C1 = exp(log_C1);
			double constProd = 1;
			double log_constProd = 0;
			for (vector<spnNode*>::iterator currentChild = prod->pChildren.begin(); currentChild< prod->pChildren.end(); currentChild++) {
				if ((*currentChild) != current) {
					constProd *= (*currentChild)->inferenceValue;
					log_constProd += (*currentChild)->log_inferenceValue;
				}
			}
			current->learning_vector[2] = log(C2) + log_constProd + parentSum->learning_vector[2];
			current->learning_vector[3] = log(C2) + log_constProd + parentSum->learning_vector[3];
			current->learning_vector[4] = log(C2) + log_constProd + parentSum->learning_vector[4];

			maximum = parentSum->learning_vector[5];
			if (log_C1 + parentSum->learning_vector[4] > maximum)
				maximum = log_C1 + parentSum->learning_vector[4];

			current->learning_vector[5] = log(exp(parentSum->learning_vector[5] - maximum) + exp(log_C1 + parentSum->learning_vector[4] - maximum)) + maximum;

			maximum = current->learning_vector[2];
			if (current->learning_vector[3] > maximum)
				maximum = current->learning_vector[3];

			current->learning_vector[0] = log(exp(current->learning_vector[2] - maximum) + exp(current->learning_vector[3] - maximum)) + maximum;
			current->learning_vector[1] = current->learning_vector[5];

			/*if (isinf(current->learning_vector[0]) || isinf(current->learning_vector[1])) {
			cout << " weightSum: " << weightSum << endl;
			cout << " Parent Sum Inference: ";
			for (spnNode* i : parentSum->pChildren) {
			cout << i->inferenceValue << "  ";
			}
			cout << endl;
			cout << " Parent Prod Inference: ";
			for (spnNode* i : prod->pChildren) {
			cout << i->inferenceValue << "  ";
			}
			cout << endl;

			exit(EXIT_FAILURE);
			}*/
		}
	}

}

void SPN::doDownwardPassContinuous(double *Instance) {
	// This needs to be fixed to work with log scale.
	for (vector<spnNode*>::iterator nodeIter = spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++) {
		spnNode* current = (*nodeIter);

		if (current->Type == "Sum" && current->ID != root) {
			spnNode* prod = current->pParents[0];
			spnNode* parentSum = prod->pParents[0];
			double C1 = 0, log_C1 = 0;
			double C2 = 0;
			double weightSum = std::accumulate(parentSum->weights.begin(), parentSum->weights.end(), 0.0);

			
			double maximum = parentSum->pChildren[0]->log_inferenceValue;
			for (const spnNode* s : parentSum->pChildren) {
				if (s->log_inferenceValue > maximum) {
					maximum = s->log_inferenceValue;
				}
			}
			
			vector<double>::iterator currentWeight = parentSum->weights.begin();
			for (vector<spnNode*>::iterator currentChild = parentSum->pChildren.begin(); currentChild< parentSum->pChildren.end(); currentChild++, currentWeight++) {
				if ((*currentChild) != prod)
					C1 += exp((*currentChild)->log_inferenceValue + log((*currentWeight) / weightSum) - maximum);
				else
					C2 = (*currentWeight) / weightSum;
			}
			log_C1 = log(C1) + maximum;
			C1 = exp(log_C1);
			double constProd = 1;
			double log_constProd = 0;
			for (vector<spnNode*>::iterator currentChild = prod->pChildren.begin(); currentChild< prod->pChildren.end(); currentChild++) {
				if ((*currentChild) != current) {
					constProd *= (*currentChild)->inferenceValue;
					log_constProd += (*currentChild)->log_inferenceValue;
				}
			}
			current->learning_vector[2] = log(C2)+log_constProd+parentSum->learning_vector[2];
			current->learning_vector[3] = log(C2)+log_constProd+parentSum->learning_vector[3];
			current->learning_vector[4] = log(C2)+log_constProd+parentSum->learning_vector[4];

			maximum = parentSum->learning_vector[5];
			if (log_C1 + parentSum->learning_vector[4] > maximum)
				maximum = log_C1 + parentSum->learning_vector[4];

			current->learning_vector[5] = log(exp(parentSum->learning_vector[5]- maximum) + exp(log_C1+parentSum->learning_vector[4]-maximum)) + maximum;

			maximum = current->learning_vector[2];
			if (current->learning_vector[3] > maximum)
				maximum = current->learning_vector[3];

			current->learning_vector[0] = log(exp(current->learning_vector[2]-maximum) + exp(current->learning_vector[3]-maximum)) + maximum;
			current->learning_vector[1] = current->learning_vector[5];

			/*if (isinf(current->learning_vector[0]) || isinf(current->learning_vector[1])) {
				cout << " weightSum: " << weightSum << endl;
				cout << " Parent Sum Inference: ";
				for (spnNode* i : parentSum->pChildren) {
					cout << i->inferenceValue << "  ";
				}
				cout << endl;
				cout << " Parent Prod Inference: ";
				for (spnNode* i : prod->pChildren) {
					cout << i->inferenceValue << "  ";
				}
				cout << endl;

				exit(EXIT_FAILURE);
			}*/
		}
	}
}
void SPN::updateWeights() {
	for(vector<spnNode*>::iterator nodeIter=spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++){
		spnNode* current= (*nodeIter);

		if(current->Type == "Sum"){
			double C1, C2;
			vector<double> weights, log_weights;
			vector<double> inference(current->weights.size());
			//vector<double> inference1(current->weights.size());
			C1 = current->learning_vector[0];
			
			C2 = current->learning_vector[1];
			weights = current->weights;
			log_weights = current->log_weights;
			double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);

			


			int i=0;
			for (vector<spnNode*>::iterator currentChild= current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++){
				inference[i] = (*currentChild)->log_inferenceValue;
				//inference1[i] = (*currentChild)->log_inferenceValue;
				i++;
			}
			
			double log_C1 = current->learning_vector[0];
			//if (C1 == 0) {
			//	log_C1 = -1000;
			//}
			double log_C2 = current->learning_vector[1];
			//if (C2 == 0) {
			//	log_C2 = -1000;
			//}
			for (int i=0; i< weights.size() ; i++){
				inference[i] = inference[i] + log_C1 + log_weights[i] - log(weightSum);
				if (isnan(inference[i]))
				{
					cout<< "inference is nan0" << endl;
					cout << "inference: " <<inference[i] <<" C1: " << C1 << " weights[i] : " << weights[i] << " weightSum: " << weightSum << endl;
					exit (EXIT_FAILURE);
				}
			}
			inference.push_back(log_C2);

			double maximum = *max_element(inference.begin(), inference.end());
			transform( inference.begin(), inference.end(), inference.begin(),std::bind2nd(std::plus<double>(), 0.0 - maximum));

			double sumInference=0;
			
			for (int i=0; i< inference.size() ; i++)
			{
				if (isnan(inference[i]))
				{
					cout<< "inference is nan1" << endl;
					for (int l=0; l<inference.size(); l++) {
						cout << inference[l] << " " ;
					}
					cout << endl;
					exit (EXIT_FAILURE);
				}
				sumInference += exp(inference[i]);
			}


			for (int i=0; i< inference.size() ; i++) {
				inference[i] = exp( inference[i] - log(sumInference) );
				if (isnan(inference[i]))
				{
					cout<< "inference is nan2" << endl;
					for (int l=0; l<inference.size(); l++) {
						cout << inference[l] << " " ;
					}
					cout << endl;
					exit (EXIT_FAILURE);
				}
			}
			
			
			vector<vector< double >> Alphas;

			for (int i = 0; i < inference.size() ; i++ ) {
				Alphas.push_back(weights);
				if ( i < inference.size() - 1 ) {
					Alphas.back()[i]++;
				}
			}

			vector<double> SumAlphas(Alphas.size());
			for (int i = 0; i < Alphas.size(); i++) {
				SumAlphas[i] = std::accumulate(Alphas[i].begin(), Alphas[i].end(), 0.0);
				if (SumAlphas[i] == 0)
					SumAlphas[i] = 0.0000001;
			}

			vector<vector<double>> M(Alphas.size(), vector<double>(Alphas[0].size() + 1));
			vector<double> MT(Alphas[0].size() + 1);
			for (int i = 0; i < MT.size(); i++)
				MT[i] = exp(-100.0);
			for (int i = 0; i < M.size(); i++) {
				for (int j = 0; j < M[i].size(); j++) {
					if (j < M[i].size() - 1)
						M[i][j] = inference[i] * Alphas[i][j] / SumAlphas[i];
					else
						M[i][j] = inference[i] * (Alphas[i][0] * (Alphas[i][0] + 1)) / (SumAlphas[i] * (SumAlphas[i] + 1));
					MT[j] += M[i][j];
				}
			}

			double scaling_factor = (MT[0] - MT.back()) / (MT.back() - (MT[0] * MT[0]));
			if ((MT[0] - MT.back()) == 0 || (MT.back() - (MT[0] * MT[0])) == 0 || scaling_factor <= 0) {
				scaling_factor = 1;
			}
			for ( int i = 0; i < current->weights.size(); i++) {
				current->weights[i] = scale * MT[i] * scaling_factor + (1 - scale) * current->weights[i];
				current->log_weights[i] = log(current->weights[i]);
				if (isnan(current->weights[i]) || current->weights[i] < 0) {
					cout << "weight is nan or negative!\n";
					cout << " i: "<< i<< " current->weights[i]: " << current->weights[i] << " Scaling factor: " << scaling_factor << endl;
					cout << " MT: " ;
					for (int l=0; l < MT.size() ; l++) {
						cout << MT[l] << " ";
					}
					cout << endl;
					exit (EXIT_FAILURE);
				}
			}
		}
	}
}
double* SPN::sampleInstanceContinuous() {
	double *Instance= new double[numberOfVariables];
	queue<spnNode*> search_queue;
	search_queue.push(spnNetwork[root]);
	while (!search_queue.empty()) {
		spnNode *current = search_queue.front();
		search_queue.pop();
		if (current->Type == "Sum" && current->mixture_node) {
			double value;
			int idx;
			//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(rand());
			std::discrete_distribution<int> distribution(current->weights.begin(), current->weights.end());
			mixture m = current->gaussian_mixture[distribution(generator)];
			std::normal_distribution<double> ndistribution(m.NG[0], pow(m.NG[3]/m.NG[2],0.5));
			value = ndistribution(generator);
			idx = current->pChildren[0]->featureIdx;
			Instance[idx] = value;
		}
		else if (current->Type == "Sum") {
			//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(rand());
			std::discrete_distribution<int> distribution(current->weights.begin(), current->weights.end());
			int ridx = distribution(generator);
			search_queue.push(current->pChildren[ridx]);
		}
		else {
			for (spnNode* n : current->pChildren) {
				search_queue.push(n);
			}
		}
	}
	return Instance;
}
vector<double*> SPN::generateData(int data_size) {
	vector<double*> data;
	for (int i = 0; i < data_size; i++) {
		double *Instance = sampleInstanceContinuous();
		data.push_back(Instance);
	}
	return data;
}
void SPN::updateWeightsContinuous() {
	for (vector<spnNode*>::iterator nodeIter = spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++) {
		spnNode* current = (*nodeIter);

		if (current->Type == "Sum" && current->mixture_node) {
			//double C1, C2;
			vector<double> weights, log_weights;
			vector<double> inference(current->weights.size());
			//vector<double> inference1(current->weights.size());
			//C1 = current->learning_vector[0];
			//C2 = current->learning_vector[1];
			weights = current->weights;
			log_weights = current->log_weights;
			double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);




			int i = 0;
			for (vector<double>::iterator current_inference = current->log_inference.begin(); current_inference< current->log_inference.end(); current_inference++) {
				inference[i] = *current_inference;// -log(weightSum);
				//inference1[i] = (*currentChild)->log_inferenceValue;
				i++;
			}

			double log_C1 =  current->learning_vector[0];
			double log_C2 =  current->learning_vector[1];

			for (int i = 0; i< weights.size(); i++) {
				inference[i] = inference[i] + log_C1 -log(weightSum);
				if (isnan(inference[i]))
				{
					cout << "inference is nan0" << endl;
					cout << "inference: " << inference[i] << " log C1: " << log_C1 << " weights[i] : " << weights[i] << " weightSum: " << weightSum << endl;
					exit(EXIT_FAILURE);
				}
			}
			inference.push_back(log_C2);

			double maximum = *max_element(inference.begin(), inference.end());
			transform(inference.begin(), inference.end(), inference.begin(), std::bind2nd(std::plus<double>(), 0.0 - maximum));

			double sumInference = 0;

			for (int i = 0; i< inference.size(); i++)
			{
				if (isnan(inference[i]))
				{
					cout << "inference is nan1" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					cout << "Log C2: " << log_C2 << endl;
					exit(EXIT_FAILURE);
				}
				sumInference += exp(inference[i]);
			}


			for (int i = 0; i< inference.size(); i++) {
				inference[i] = exp(inference[i] - log(sumInference));
				if (isnan(inference[i]))
				{
					cout << "inference is nan2" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}

			// here the vector is ready to work with C2 + C1 * Sum (alphai/sum(alpha) * ci * dir() * prod(NG()))

			vector<vector<double>> GMoments(current->gaussian_mixture.size(),vector<double>(4)); // 4 moments are needed per Gaussian
			for (i = 0; i < GMoments.size(); i++) {
				double moment_updated, moment;
				double inferenceSum = std::accumulate(inference.begin(), inference.end(), 0.0);

				double cinference = inference[i];
				// M0
				moment = current->gaussian_mixture[i].NG[0];
				moment_updated = current->updated_gaussian_mixture[i].NG[0];
				GMoments[i][0] = cinference * moment_updated + (1 - cinference) * moment;

				// M1
				moment = current->gaussian_mixture[i].NG[2] / current->gaussian_mixture[i].NG[3];
				moment_updated = current->updated_gaussian_mixture[i].NG[2] / current->updated_gaussian_mixture[i].NG[3];
				GMoments[i][1] = cinference * moment_updated + (1 - cinference) * moment;

				// M2
				moment = current->gaussian_mixture[i].NG[2] * (current->gaussian_mixture[i].NG[2] + 1) / pow(current->gaussian_mixture[i].NG[3],2);
				moment_updated = current->updated_gaussian_mixture[i].NG[2] * (current->updated_gaussian_mixture[i].NG[2] + 1) / pow(current->updated_gaussian_mixture[i].NG[3],2);
				GMoments[i][2] = cinference * moment_updated + (1 - cinference) * moment;

				// M3
				moment = pow(current->gaussian_mixture[i].NG[0],2) + current->gaussian_mixture[i].NG[3] /  (current->gaussian_mixture[i].NG[1] * (current->gaussian_mixture[i].NG[2]-1));
				moment_updated = pow(current->updated_gaussian_mixture[i].NG[0], 2) + current->updated_gaussian_mixture[i].NG[3] / (current->updated_gaussian_mixture[i].NG[1] * (current->updated_gaussian_mixture[i].NG[2] - 1));
				//moment = 1/ current->gaussian_mixture[i].NG[1] + pow(current->gaussian_mixture[i].NG[2],2) * current->gaussian_mixture[i].NG[2] / current->gaussian_mixture[i].NG[3];
				//moment_updated = 1 / current->updated_gaussian_mixture[i].NG[1] + pow(current->updated_gaussian_mixture[i].NG[2], 2) * current->updated_gaussian_mixture[i].NG[2] / current->updated_gaussian_mixture[i].NG[3];
				GMoments[i][3] = cinference * moment_updated + (1 - cinference) * moment;


				// updating mixture
				// Mu
				current->gaussian_mixture[i].NG[0] = GMoments[i][0];

				// Alpha and Beta
				if (GMoments[i][2] - pow(GMoments[i][1], 2) <= 0) {

					cout << "Warning: M2 - M1^2 <= 0\n";
					cout << "GMoments[i][2] : " << GMoments[i][2] << " GMoments[i][1]: " << GMoments[i][1] << " inferenceSum: "<< inferenceSum << endl;
					current->gaussian_mixture[i].NG[2] = 2;
					current->gaussian_mixture[i].NG[3] = 2;
				}
				else {
					current->gaussian_mixture[i].NG[2] = pow(GMoments[i][1],2) / (GMoments[i][2] - pow(GMoments[i][1], 2));
					current->gaussian_mixture[i].NG[3] = GMoments[i][1] / (GMoments[i][2] - pow(GMoments[i][1], 2));

				}


				// K
				if (((current->gaussian_mixture[i].NG[2] - 1) * (GMoments[i][3] - pow(GMoments[i][0], 2))) <= 0) {
					cout << "Warning: Bnew / (alpha_new -1) / (M3 - M0^2) <= 0\n";
					current->gaussian_mixture[i].NG[1] =  0.001;
				}
				else {
					current->gaussian_mixture[i].NG[1] = current->gaussian_mixture[i].NG[3] / ((current->gaussian_mixture[i].NG[2] - 1) * (GMoments[i][3]-pow(GMoments[i][0],2)));
				}

			}


			/*i = 0;
			for (vector<double>::iterator current_inference = current->log_inference.begin(); current_inference< current->log_inference.end(); current_inference++) {
				inference[i] = *current_inference;// -log(weightSum);
												  //inference1[i] = (*currentChild)->log_inferenceValue;
				i++;
			}

			log_C1 = 0; // current->learning_vector[0];
			log_C2 = -1000; // current->learning_vector[1];

			for (int i = 0; i< weights.size(); i++) {
				inference[i] = inference[i] + log_C1 - log(weightSum);
				if (isnan(inference[i]))
				{
					cout << "inference is nan0" << endl;
					cout << "inference: " << inference[i] << " log C1: " << log_C1 << " weights[i] : " << weights[i] << " weightSum: " << weightSum << endl;
					exit(EXIT_FAILURE);
				}
			}
			inference.back() = log_C2;

			maximum = *max_element(inference.begin(), inference.end());
			transform(inference.begin(), inference.end(), inference.begin(), std::bind2nd(std::plus<double>(), 0.0 - maximum));

			sumInference = 0;

			for (int i = 0; i< inference.size(); i++)
			{
				if (isnan(inference[i]))
				{
					cout << "inference is nan1" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					cout << "Log C2: " << log_C2 << endl;
					exit(EXIT_FAILURE);
				}
				sumInference += exp(inference[i]);
			}


			for (int i = 0; i< inference.size(); i++) {
				inference[i] = exp(inference[i] - log(sumInference));
				if (isnan(inference[i]))
				{
					cout << "inference is nan2" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			} */

			vector<vector< double >> Alphas;

			for (int i = 0; i < inference.size(); i++) {
				Alphas.push_back(weights);
				if (i < inference.size() - 1) {
					Alphas.back()[i]++;
				}
			}

			vector<double> SumAlphas(Alphas.size());
			for (int i = 0; i < Alphas.size(); i++) {
				SumAlphas[i] = std::accumulate(Alphas[i].begin(), Alphas[i].end(), 0.0);
				if (SumAlphas[i] == 0)
					SumAlphas[i] = 0.0000001;
			}

			vector<vector<double>> M(Alphas.size(), vector<double>(Alphas[0].size() + 1));
			vector<double> MT(Alphas[0].size() + 1);
			for (int i = 0; i < MT.size(); i++)
				MT[i] = exp(-100.0);
			for (int i = 0; i < M.size(); i++) {
				for (int j = 0; j < M[i].size(); j++) {
					if (j < M[i].size() - 1)
						M[i][j] = inference[i] * Alphas[i][j] / SumAlphas[i];
					else
						M[i][j] = inference[i] * (Alphas[i][0] * (Alphas[i][0] + 1)) / (SumAlphas[i] * (SumAlphas[i] + 1));
					MT[j] += M[i][j];
				}
			}

			double scaling_factor = (MT[0] - MT.back()) / (MT.back() - (MT[0] * MT[0]));
			if ((MT[0] - MT.back()) == 0 || (MT.back() - (MT[0] * MT[0])) == 0 || scaling_factor <= 0) {
				scaling_factor = 1;
			}
			for (int i = 0; i < current->weights.size(); i++) {
				current->weights[i] = scale * MT[i] * scaling_factor + (1 - scale) * current->weights[i];
				current->log_weights[i] = log(current->weights[i]);
				if (isnan(current->weights[i]) || current->weights[i] < 0) {
					cout << "weight is nan or negative!\n";
					cout << " i: " << i << " current->weights[i]: " << current->weights[i] << " Scaling factor: " << scaling_factor << endl;
					cout << " MT: ";
					for (int l = 0; l < MT.size(); l++) {
						cout << MT[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}



		} else if (current->Type == "Sum") {
			double log_C1, log_C2;
			vector<double> weights, log_weights;
			vector<double> inference(current->weights.size());
			//vector<double> inference1(current->weights.size());
			log_C1 = current->learning_vector[0];
			log_C2 = current->learning_vector[1];
			weights = current->weights;
			log_weights = current->log_weights;
			double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);




			int i = 0;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++) {
				inference[i] = (*currentChild)->log_inferenceValue;
				//inference1[i] = (*currentChild)->log_inferenceValue;
				i++;
			}


			for (int i = 0; i< weights.size(); i++) {
				inference[i] = inference[i] + log_C1 + log_weights[i] - log(weightSum);
				if (isnan(inference[i]))
				{
					cout << "inference is nan0" << endl;
					cout << "inference: " << inference[i] << " C1: " << log_C1 << " weights[i] : " << weights[i] << " weightSum: " << weightSum << endl;
					exit(EXIT_FAILURE);
				}
			}
			inference.push_back(log_C2);

			double maximum = *max_element(inference.begin(), inference.end());
			transform(inference.begin(), inference.end(), inference.begin(), std::bind2nd(std::plus<double>(), 0.0 - maximum));

			double sumInference = 0;

			for (int i = 0; i< inference.size(); i++)
			{
				if (isnan(inference[i]))
				{
					cout << "inference is nan1" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}

					cout << endl;
					cout << "log_C1: " << log_C1 << " log_C2: " << log_C2 << endl;
					exit(EXIT_FAILURE);
				}
				sumInference += exp(inference[i]);
			}


			for (int i = 0; i< inference.size(); i++) {
				inference[i] = exp(inference[i] - log(sumInference));
				if (isnan(inference[i]))
				{
					cout << "inference is nan2" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}


			vector<vector< double >> Alphas;

			for (int i = 0; i < inference.size(); i++) {
				Alphas.push_back(weights);
				if (i < inference.size() - 1) {
					Alphas.back()[i]++;
				}
			}

			vector<double> SumAlphas(Alphas.size());
			for (int i = 0; i < Alphas.size(); i++) {
				SumAlphas[i] = std::accumulate(Alphas[i].begin(), Alphas[i].end(), 0.0);
				if (SumAlphas[i] == 0)
					SumAlphas[i] = 0.0000001;
			}

			vector<vector<double>> M(Alphas.size(), vector<double>(Alphas[0].size() + 1));
			vector<double> MT(Alphas[0].size() + 1);
			for (int i = 0; i < MT.size(); i++)
				MT[i] = exp(-100.0);
			for (int i = 0; i < M.size(); i++) {
				for (int j = 0; j < M[i].size(); j++) {
					if (j < M[i].size() - 1)
						M[i][j] = inference[i] * Alphas[i][j] / SumAlphas[i];
					else
						M[i][j] = inference[i] * (Alphas[i][0] * (Alphas[i][0] + 1)) / (SumAlphas[i] * (SumAlphas[i] + 1));
					MT[j] += M[i][j];
				}
			}

			double scaling_factor = (MT[0] - MT.back()) / (MT.back() - (MT[0] * MT[0]));
			if ((MT[0] - MT.back()) == 0 || (MT.back() - (MT[0] * MT[0])) == 0 || scaling_factor <= 0) {
				scaling_factor = 1;
			}
			for (int i = 0; i < current->weights.size(); i++) {
				current->weights[i] = scale * MT[i] * scaling_factor + (1 - scale) * current->weights[i];
				current->log_weights[i] = log(current->weights[i]);
				if (isnan(current->weights[i]) || current->weights[i] < 0) {
					cout << "weight is nan or negative!\n";
					cout << " i: " << i << " current->weights[i]: " << current->weights[i] << " Scaling factor: " << scaling_factor << endl;
					cout << " MT: ";
					for (int l = 0; l < MT.size(); l++) {
						cout << MT[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}
		}
	}
}

void SPN::updateWeightsContinuous1() {
	for (vector<spnNode*>::iterator nodeIter = spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++) {
		spnNode* current = (*nodeIter);

		if (current->Type == "Sum" && current->mixture_node) {
			//double C1, C2;
			vector<double> weights, log_weights;
			vector<double> inference(current->weights.size());
			//vector<double> inference1(current->weights.size());
			//C1 = current->learning_vector[0];
			//C2 = current->learning_vector[1];
			weights = current->weights;
			log_weights = current->log_weights;
			double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);




			int i = 0;
			for (vector<double>::iterator current_inference = current->log_inference.begin(); current_inference< current->log_inference.end(); current_inference++) {
				inference[i] = *current_inference;// -log(weightSum);
												  //inference1[i] = (*currentChild)->log_inferenceValue;
				i++;
			}

			double log_C1 =  current->learning_vector[0];
			double log_C2 =  current->learning_vector[1];

			for (int i = 0; i< weights.size(); i++) {
				inference[i] = inference[i] + log_C1 - log(weightSum);
				if (isnan(inference[i]))
				{
					cout << "inference is nan0" << endl;
					cout << "inference: " << inference[i] << " log C1: " << log_C1 << " weights[i] : " << weights[i] << " weightSum: " << weightSum << endl;
					exit(EXIT_FAILURE);
				}
			}
			inference.push_back(log_C2);

			double maximum = *max_element(inference.begin(), inference.end());
			transform(inference.begin(), inference.end(), inference.begin(), std::bind2nd(std::plus<double>(), 0.0 - maximum));

			double sumInference = 0;

			for (int i = 0; i< inference.size(); i++)
			{
				if (isnan(inference[i]))
				{
					cout << "inference is nan1" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					cout << "Log C2: " << log_C2 << endl;
					exit(EXIT_FAILURE);
				}
				sumInference += exp(inference[i]);
			}


			for (int i = 0; i< inference.size(); i++) {
				inference[i] = exp(inference[i] - log(sumInference));
				if (isnan(inference[i]))
				{
					cout << "inference is nan2" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}

			vector<vector< double >> Alphas;

			for (int i = 0; i < inference.size(); i++) {
				Alphas.push_back(weights);
				if (i < inference.size() - 1) {
					Alphas.back()[i]++;
				}
			}

			vector<double> SumAlphas(Alphas.size());
			for (int i = 0; i < Alphas.size(); i++) {
				SumAlphas[i] = std::accumulate(Alphas[i].begin(), Alphas[i].end(), 0.0);
				if (SumAlphas[i] == 0)
					SumAlphas[i] = 0.0000001;
			}

			vector<vector<double>> M(Alphas.size(), vector<double>(Alphas[0].size() + 1));
			vector<double> MT(Alphas[0].size() + 1);
			for (int i = 0; i < MT.size(); i++)
				MT[i] = exp(-100.0);
			for (int i = 0; i < M.size(); i++) {
				for (int j = 0; j < M[i].size(); j++) {
					if (j < M[i].size() - 1)
						M[i][j] = inference[i] * Alphas[i][j] / SumAlphas[i];
					else
						M[i][j] = inference[i] * (Alphas[i][0] * (Alphas[i][0] + 1)) / (SumAlphas[i] * (SumAlphas[i] + 1));
					MT[j] += M[i][j];
				}
			}

			double scaling_factor = (MT[0] - MT.back()) / (MT.back() - (MT[0] * MT[0]));
			if ((MT[0] - MT.back()) == 0 || (MT.back() - (MT[0] * MT[0])) == 0 || scaling_factor <= 0) {
				scaling_factor = 1;
			}
			for (int i = 0; i < current->weights.size(); i++) {
				current->weights[i] = scale * MT[i] * scaling_factor + (1 - scale) * current->weights[i];
				current->log_weights[i] = log(current->weights[i]);
				if (isnan(current->weights[i]) || current->weights[i] < 0) {
					cout << "weight is nan or negative!\n";
					cout << " i: " << i << " current->weights[i]: " << current->weights[i] << " Scaling factor: " << scaling_factor << endl;
					cout << " MT: ";
					for (int l = 0; l < MT.size(); l++) {
						cout << MT[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}



		}
		else if (current->Type == "Sum") {
			double log_C1, log_C2;
			vector<double> weights, log_weights;
			vector<double> inference(current->weights.size());
			//vector<double> inference1(current->weights.size());
			log_C1 = current->learning_vector[0];
			log_C2 = current->learning_vector[1];
			weights = current->weights;
			log_weights = current->log_weights;
			double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);




			int i = 0;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++) {
				inference[i] = (*currentChild)->log_inferenceValue;
				//inference1[i] = (*currentChild)->log_inferenceValue;
				i++;
			}


			for (int i = 0; i< weights.size(); i++) {
				inference[i] = inference[i] + log_C1 + log_weights[i] - log(weightSum);
				if (isnan(inference[i]))
				{
					cout << "inference is nan0" << endl;
					cout << "inference: " << inference[i] << " C1: " << log_C1 << " weights[i] : " << weights[i] << " weightSum: " << weightSum << endl;
					exit(EXIT_FAILURE);
				}
			}
			inference.push_back(log_C2);

			double maximum = *max_element(inference.begin(), inference.end());
			transform(inference.begin(), inference.end(), inference.begin(), std::bind2nd(std::plus<double>(), 0.0 - maximum));

			double sumInference = 0;

			for (int i = 0; i< inference.size(); i++)
			{
				if (isnan(inference[i]))
				{
					cout << "inference is nan1" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}

					cout << endl;
					cout << "log_C1: " << log_C1 << " log_C2: " << log_C2 << endl;
					exit(EXIT_FAILURE);
				}
				sumInference += exp(inference[i]);
			}


			for (int i = 0; i< inference.size(); i++) {
				inference[i] = exp(inference[i] - log(sumInference));
				if (isnan(inference[i]))
				{
					cout << "inference is nan2" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}


			vector<vector< double >> Alphas;

			for (int i = 0; i < inference.size(); i++) {
				Alphas.push_back(weights);
				if (i < inference.size() - 1) {
					Alphas.back()[i]++;
				}
			}

			vector<double> SumAlphas(Alphas.size());
			for (int i = 0; i < Alphas.size(); i++) {
				SumAlphas[i] = std::accumulate(Alphas[i].begin(), Alphas[i].end(), 0.0);
				if (SumAlphas[i] == 0)
					SumAlphas[i] = 0.0000001;
			}

			vector<vector<double>> M(Alphas.size(), vector<double>(Alphas[0].size() + 1));
			vector<double> MT(Alphas[0].size() + 1);
			for (int i = 0; i < MT.size(); i++)
				MT[i] = exp(-100.0);
			for (int i = 0; i < M.size(); i++) {
				for (int j = 0; j < M[i].size(); j++) {
					if (j < M[i].size() - 1)
						M[i][j] = inference[i] * Alphas[i][j] / SumAlphas[i];
					else
						M[i][j] = inference[i] * (Alphas[i][0] * (Alphas[i][0] + 1)) / (SumAlphas[i] * (SumAlphas[i] + 1));
					MT[j] += M[i][j];
				}
			}

			double scaling_factor = (MT[0] - MT.back()) / (MT.back() - (MT[0] * MT[0]));
			if ((MT[0] - MT.back()) == 0 || (MT.back() - (MT[0] * MT[0])) == 0 || scaling_factor <= 0) {
				scaling_factor = 1;
			}
			for (int i = 0; i < current->weights.size(); i++) {
				current->weights[i] = scale * MT[i] * scaling_factor + (1 - scale) * current->weights[i];
				current->log_weights[i] = log(current->weights[i]);
				if (isnan(current->weights[i]) || current->weights[i] < 0) {
					cout << "weight is nan or negative!\n";
					cout << " i: " << i << " current->weights[i]: " << current->weights[i] << " Scaling factor: " << scaling_factor << endl;
					cout << " MT: ";
					for (int l = 0; l < MT.size(); l++) {
						cout << MT[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}
		}
	}
}

void SPN::updateWeightsContinuousMultivariate() {
	for (vector<spnNode*>::iterator nodeIter = spnNetwork.begin(); nodeIter<spnNetwork.end(); nodeIter++) {
		spnNode* current = (*nodeIter);

		if (current->Type == "Sum" && current->mixture_node) {
			//double C1, C2;
			vector<double> weights, log_weights;
			vector<double> inference(2);
			//vector<double> inference1(current->weights.size());
			//C1 = current->learning_vector[0];
			//C2 = current->learning_vector[1];
			weights = current->weights;
			log_weights = current->log_weights;
			double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);




			inference[0] = current->log_inferenceValue;

			double log_C1 = current->learning_vector[0];
			double log_C2 = current->learning_vector[1];

			inference[0] += log_C1;
			inference[1] = log_C2;

			double maximum = *max_element(inference.begin(), inference.end());
			transform(inference.begin(), inference.end(), inference.begin(), std::bind2nd(std::plus<double>(), 0.0 - maximum));

			double sumInference = 0;

			for (int i = 0; i< inference.size(); i++)
			{
				if (isnan(inference[i]))
				{
					cout << "inference is nan1" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					cout << "Log C2: " << log_C2 << endl;
					exit(EXIT_FAILURE);
				}
				sumInference += exp(inference[i]);
			}


			for (int i = 0; i< inference.size(); i++) {
				inference[i] = exp(inference[i] - log(sumInference));
				if (isnan(inference[i]))
				{
					cout << "inference is nan2" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}

			// here the vector is ready to work with C2 + C1 * Sum (alphai/sum(alpha) * ci * dir() * prod(NG()))
			struct moments {
				Eigen::MatrixXd Wishart1;
				Eigen::MatrixXd Wishart2;
				Eigen::MatrixXd Normal1;
				Eigen::MatrixXd Normal2;
			};
			moments GMoments;
			int d = current->scope.size();
			double moment_updated, moment;
			double inferenceSum = std::accumulate(inference.begin(), inference.end(), 0.0);

			double k = inference[0];
			// Wishart1
			GMoments.Wishart1.resize(d, d);
			GMoments.Wishart1 = k * (current->mmixture.v+1) * current->updated_mmixture.W + (1 - k) * current->mmixture.v * current->mmixture.W;

			// Wishart2
			GMoments.Wishart2.resize(d, d);
			Eigen::MatrixXd Diag, DiagS;
			Diag = current->mmixture.W.diagonal(0);

			DiagS = current->updated_mmixture.W.diagonal(0);
			GMoments.Wishart2 = k * (current->mmixture.v + 1) * (current->updated_mmixture.W.cwiseProduct(current->updated_mmixture.W) + DiagS*DiagS.transpose()) + (1 - k) * current->mmixture.v * (current->mmixture.W.cwiseProduct(current->mmixture.W) + Diag * Diag.transpose());


			// Normal1
			GMoments.Normal1.resize(1, d);
			GMoments.Normal1 = k * current->updated_mmixture.mu + (1 - k) * current->mmixture.mu;

			// Normal2
			GMoments.Normal2.resize(d, d);
			GMoments.Normal2 = k * ((1 / (current->updated_mmixture.k * (current->updated_mmixture.v - d - 1))) * current->updated_mmixture.W + current->updated_mmixture.mu.transpose() * current->updated_mmixture.mu) + (1-k)   *(  1 / (current->mmixture.k * (current->mmixture.v - d - 1)) * current->mmixture.W + current->mmixture.mu.transpose() * current->mmixture.mu);


			// updating mixture
			// Mu
			current->mmixture.mu = GMoments.Normal1;

			// v
			current->mmixture.v = 2 * GMoments.Wishart1(0, 0) * GMoments.Wishart1(0, 0) / GMoments.Wishart2(0, 0);
			
			// W
			current->mmixture.W = (1/current->mmixture.v) * GMoments.Wishart1;

			// k
			Eigen::MatrixXd Kappa(d, d);
			Eigen::MatrixXd Normal2MuMu(d, d), Normal2MuMuInverse(d,d);
			Normal2MuMu= GMoments.Normal2 - current->mmixture.mu.transpose() * current->mmixture.mu;
			Normal2MuMuInverse = Normal2MuMu.inverse();
			Kappa = (1 / (current->mmixture.v -d  -1)) * current->mmixture.W * Normal2MuMuInverse;
			Eigen::EigenSolver<Eigen::MatrixXd> es(Kappa);
			Eigen::VectorXcd ev = es.eigenvalues();
			current->mmixture.k = 0.000000001;
			for (int i = 0; i < ev.rows(); i++) {
				if (ev(i).real() > current->mmixture.k) {
					current->mmixture.k = ev(i).real();
				}
			}
		}
		else if (current->Type == "Sum") {
			double log_C1, log_C2;
			vector<double> weights, log_weights;
			vector<double> inference(current->weights.size());
			//vector<double> inference1(current->weights.size());
			log_C1 = current->learning_vector[0];
			log_C2 = current->learning_vector[1];
			weights = current->weights;
			log_weights = current->log_weights;
			double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);




			int i = 0;
			for (vector<spnNode*>::iterator currentChild = current->pChildren.begin(); currentChild< current->pChildren.end(); currentChild++) {
				inference[i] = (*currentChild)->log_inferenceValue;
				//inference1[i] = (*currentChild)->log_inferenceValue;
				i++;
			}


			for (int i = 0; i< weights.size(); i++) {
				inference[i] = inference[i] + log_C1 + log_weights[i] - log(weightSum);
				if (isnan(inference[i]))
				{
					cout << "inference is nan0" << endl;
					cout << "inference: " << inference[i] << " C1: " << log_C1 << " weights[i] : " << weights[i] << " weightSum: " << weightSum << endl;
					exit(EXIT_FAILURE);
				}
			}
			inference.push_back(log_C2);

			double maximum = *max_element(inference.begin(), inference.end());
			transform(inference.begin(), inference.end(), inference.begin(), std::bind2nd(std::plus<double>(), 0.0 - maximum));

			double sumInference = 0;

			for (int i = 0; i< inference.size(); i++)
			{
				if (isnan(inference[i]))
				{
					cout << "inference is nan1" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}

					cout << endl;
					cout << "log_C1: " << log_C1 << " log_C2: " << log_C2 << endl;
					exit(EXIT_FAILURE);
				}
				sumInference += exp(inference[i]);
			}


			for (int i = 0; i< inference.size(); i++) {
				inference[i] = exp(inference[i] - log(sumInference));
				if (isnan(inference[i]))
				{
					cout << "inference is nan2" << endl;
					for (int l = 0; l<inference.size(); l++) {
						cout << inference[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}


			vector<vector< double >> Alphas;

			for (int i = 0; i < inference.size(); i++) {
				Alphas.push_back(weights);
				if (i < inference.size() - 1) {
					Alphas.back()[i]++;
				}
			}

			vector<double> SumAlphas(Alphas.size());
			for (int i = 0; i < Alphas.size(); i++) {
				SumAlphas[i] = std::accumulate(Alphas[i].begin(), Alphas[i].end(), 0.0);
				if (SumAlphas[i] == 0)
					SumAlphas[i] = 0.0000001;
			}

			vector<vector<double>> M(Alphas.size(), vector<double>(Alphas[0].size() + 1));
			vector<double> MT(Alphas[0].size() + 1);
			for (int i = 0; i < MT.size(); i++)
				MT[i] = exp(-100.0);
			for (int i = 0; i < M.size(); i++) {
				for (int j = 0; j < M[i].size(); j++) {
					if (j < M[i].size() - 1)
						M[i][j] = inference[i] * Alphas[i][j] / SumAlphas[i];
					else
						M[i][j] = inference[i] * (Alphas[i][0] * (Alphas[i][0] + 1)) / (SumAlphas[i] * (SumAlphas[i] + 1));
					MT[j] += M[i][j];
				}
			}

			double scaling_factor = (MT[0] - MT.back()) / (MT.back() - (MT[0] * MT[0]));
			if ((MT[0] - MT.back()) == 0 || (MT.back() - (MT[0] * MT[0])) == 0 || scaling_factor <= 0) {
				scaling_factor = 1;
			}
			for (int i = 0; i < current->weights.size(); i++) {
				current->weights[i] = scale * MT[i] * scaling_factor + (1 - scale) * current->weights[i];
				current->log_weights[i] = log(current->weights[i]);
				if (isnan(current->weights[i]) || current->weights[i] < 0) {
					cout << "weight is nan or negative!\n";
					cout << " i: " << i << " current->weights[i]: " << current->weights[i] << " Scaling factor: " << scaling_factor << endl;
					cout << " MT: ";
					for (int l = 0; l < MT.size(); l++) {
						cout << MT[l] << " ";
					}
					cout << endl;
					exit(EXIT_FAILURE);
				}
			}
		}
	}
}
void SPN::doMomentMatchingOnInstance(int* Instance) {
	spnNetwork[root]->learning_vector[0] = 0;
	spnNetwork[root]->learning_vector[1] = -1000;
	spnNetwork[root]->learning_vector[2] = 0;
	spnNetwork[root]->learning_vector[3] = -1000;
	spnNetwork[root]->learning_vector[4] = 0;
	spnNetwork[root]->learning_vector[5] = -1000;
	doUpwardPass(Instance);
	doDownwardPass(Instance);
	updateWeights();
}

void SPN::doMomentMatchingOnInstanceContinuous(double* Instance) {
	spnNetwork[root]->learning_vector[0] = 0;
	spnNetwork[root]->learning_vector[1] = -1000;
	spnNetwork[root]->learning_vector[2] = 0;
	spnNetwork[root]->learning_vector[3] = -1000;
	spnNetwork[root]->learning_vector[4] = 0;
	spnNetwork[root]->learning_vector[5] = -1000;
	doUpwardPassContinuous(Instance);
	doDownwardPassContinuous(Instance);
	updateWeightsContinuous();
}

void SPN::doMomentMatchingOnInstanceContinuousMultivariate(double* Instance) {
	spnNetwork[root]->learning_vector[0] = 0;
	spnNetwork[root]->learning_vector[1] = -1000;
	spnNetwork[root]->learning_vector[2] = 0;
	spnNetwork[root]->learning_vector[3] = -1000;
	spnNetwork[root]->learning_vector[4] = 0;
	spnNetwork[root]->learning_vector[5] = -1000;
	doUpwardPassContinuousMultivariate(Instance);
	doDownwardPassContinuous(Instance);
	updateWeightsContinuousMultivariate();
}

void SPN::doMomentMatchingOnInstanceContinuous1(double* Instance) {
	spnNetwork[root]->learning_vector[0] = 0;
	spnNetwork[root]->learning_vector[1] = -1000;
	spnNetwork[root]->learning_vector[2] = 0;
	spnNetwork[root]->learning_vector[3] = -1000;
	spnNetwork[root]->learning_vector[4] = 0;
	spnNetwork[root]->learning_vector[5] = -1000;
	doUpwardPassContinuous1(Instance);
	doDownwardPassContinuous(Instance);
	updateWeightsContinuous1();
}
void SPN::doMomentMatching(std::vector<int*> data) {
	//std::srand ( unsigned ( std::time(0) ) );
	std::random_shuffle( data.begin(), data.end() );
	for (int i=0; i<data.size(); i++) {
		if (i%10000 == 0) {
			cout << "Instance #: " << i << " of: " << data.size() << endl;
		}
		scale = 1 ; // 0.1 / (double) (i + 1);
		doMomentMatchingOnInstance(data[i]);
	}
}

void SPN::doMomentMatchingContinuous(std::vector<double*> data) {
	//std::srand(unsigned(std::time(0)));
	std::random_shuffle(data.begin(), data.end());
	for (int i = 0; i<data.size(); i++) {
		if (i % 1000 == 0) {
			cout << "Instance #: " << i << " of: " << data.size() << endl;
		}
		scale = 1; // 0.1 / (double) (i + 1);
		doMomentMatchingOnInstanceContinuous(data[i]);
	}

}

void SPN::doMomentMatchingContinuousMultivariate(std::vector<double*> data) {
	//std::srand(unsigned(std::time(0)));
	std::random_shuffle(data.begin(), data.end());
	for (int i = 0; i<data.size(); i++) {
		if (i % 1000 == 0) {
			cout << "Instance #: " << i << " of: " << data.size() << endl;
		}
		scale = 1; // 0.1 / (double) (i + 1);
		doMomentMatchingOnInstanceContinuousMultivariate(data[i]);
	}

}

void SPN::doMomentMatchingOnUCI(string ifile) {
	ifstream uci_file(ifile);
	int number_of_documents, number_of_words, total_words, temp;
	uci_file >> number_of_documents >> number_of_words >> total_words;
	int *vec = new int[number_of_words];
	
	double current_document;
	int word;
	uci_file >> current_document;
	for (double i=0; i<number_of_documents; i++) {
		if ((int) i %1000 == 0 ) {
			cout << "Processing instance: " << i << " of:" << number_of_documents<<endl;
		}
		for (int j=0; j < number_of_words; j++) {
			vec[j] = 0;
		}
		while (current_document == i + 1 && !uci_file.eof()) {
			uci_file >> word;
			vec[ word - 1] = 1;
			uci_file >> temp;
			uci_file >> current_document;
		}
		scale = 1 ; // 0.1 / (double) (i + 1);
		doMomentMatchingOnInstance(vec);
	}
	uci_file.close();
}
void SPN::convertUciDataset(string ifile, string ofile) {
	ifstream uci_file(ifile);
	ofstream output_file(ofile);
	
	int number_of_documents, number_of_words, total_words, temp;
	uci_file >> number_of_documents >> number_of_words >> total_words;
	int *vec = new int[number_of_words];
	
	double current_document;
	int word;
	uci_file >> current_document;
	for (double i=0; i<number_of_documents; i++) {
		if ((int) i %1000 == 0 ) {
			cout << "Processing instance: " << i << " of:" << number_of_documents<<endl;
		}
		for (int j=0; j < number_of_words; j++) {
			vec[j] = 0;
		}
		while (current_document == i + 1 && !uci_file.eof()) {
			uci_file >> word;
			vec[ word - 1] = 1;
			uci_file >> temp;
			uci_file >> current_document;
		}
		for (int j=0; j < number_of_words - 1; j++) {
			output_file << vec[j] << ",";
		}
		output_file << vec[number_of_words - 1] << endl;
	}
	uci_file.close();
	output_file.close();
}

void SPN::generateTrainTestUciDatasets(string ifile, string trainfile, string testfile, int number_of_partitions) {
	double percentage_train = 0.75;
	int train_size = 0, test_size = 0;
	int current_train = 0, current_test = 0;
	bool train;
	ifstream uci_file(ifile);
	ofstream training(trainfile);
	ofstream testing(testfile);
	
	int number_of_documents, number_of_words, total_words, temp;
	uci_file >> number_of_documents >> number_of_words >> total_words;

	vector<bool> documents(number_of_documents);
	for (int i=0; i < number_of_documents; i++) {
		documents[i] = false;
		if ( (rand() % 1000) / 1000.0 <= percentage_train) {
			documents[i] = true;
			train_size++;
		} else {
			test_size++;
		}
	}
	training << train_size << endl << number_of_words << endl << total_words << endl;
	testing << test_size << endl << number_of_words << endl << total_words << endl;


	vector<ofstream> training_parititions(number_of_partitions);
	vector<int> training_size(number_of_partitions);
	for(int i=0; i<number_of_partitions; i++) {
		training_parititions[i].open(trainfile + "." + std::to_string((long long)i));
		training_size[i] = train_size / number_of_partitions;
		if ( i == number_of_partitions - 1) {
			training_size[i] = train_size / number_of_partitions + train_size % number_of_partitions;
		}
		training_parititions[i] << training_size[i] << endl << number_of_words << endl << total_words << endl;
	}
	int  current_paritition = 0, current_document_parititions = 0;
	int *vec = new int[number_of_words];
	double current_document;
	int word;
	uci_file >> current_document;
	for (double i=0; i<number_of_documents; i++) {
		train = documents[i];
		while (current_document == i + 1 && !uci_file.eof()) {
			uci_file >> word;
			uci_file >> temp;
			if (train) {
				training << current_train + 1 << " " << word << " " << temp << endl;
				if (current_document_parititions >= training_size[current_paritition]) {
					current_paritition++;
					current_document_parititions = 0;
				}
				training_parititions[current_paritition] << current_document_parititions + 1 << " " << word << " " << temp << endl;
			} else {
				testing << current_test + 1 << " " << word << " " << temp << endl;
			}
			int temp = current_document;
			uci_file >> current_document;
			if (current_document != temp) {
				if (train) {
					current_train++;
					current_document_parititions++;
				} else {
					current_test++;
				}
			}
		}
	}
	for (int i=0; i < number_of_partitions; i++) {
		training_parititions[i].close();
	}
	uci_file.close();
	training.close();
	testing.close();
}

void SPN::generateRandomStructure(int Depth, int sum_max_num_children) {
	int current_id = 0;
	spnNode* newNode;
	newNode = new spnNode;
	newNode->ID = current_id++;
	root = 0;
	newNode->Type = "Sum";
	newNode->Depth = Depth;
	for (int i=0; i < numberOfVariables; i++) {
		newNode->scope.push_back(i);
	}
	spnNetwork.push_back(newNode);
	queue<spnNode *> search_queue;
	search_queue.push(newNode);

	// adding all leaf nodes
	for (int i = 0; i < numberOfVariables; i++) {
		for (int j = 0; j < numberOfValues; j++){
			newNode = new spnNode;
			newNode->ID = current_id++;
			newNode->Type = "Var";
			newNode->featureIdx = i;
			newNode->ValueIdx = j;
			spnNetwork.push_back(newNode);
			leaves[i][j] = newNode;
		}
	}
	while(!search_queue.empty()) {
		spnNode *current_node = search_queue.front();
		search_queue.pop();
		if (current_node->Type == "Sum" && current_node->scope.size() > 1) {
			int number_of_children = std::min(pow((double)current_node->scope.size(), (1- 1/(double)current_node->Depth)), (double) sum_max_num_children);

			for (int i=0; i < number_of_children; i++) {
				newNode = new spnNode;
				newNode->ID = current_id++;
				spnNetwork.push_back(newNode);
				search_queue.push(newNode);

				newNode->Type = "Product";
				newNode->parents.push_back(current_node->ID);
				newNode->pParents.push_back(current_node);
				newNode->Depth = current_node->Depth;
				newNode->scope = current_node->scope;

				current_node->children.push_back(newNode->ID);
				current_node->pChildren.push_back(newNode);
				current_node->weights.push_back(0.0);
				current_node->log_weights.push_back(0.0);
				current_node->log_inference.push_back(0.0);
			}
			current_node->scope.clear();
		} else if (current_node->Type == "Sum" && current_node->scope.size() == 1) {
			int current_var = current_node->scope[0];
			for (int i=0; i < numberOfValues; i++) {
				current_node->children.push_back(leaves[current_var][i]->ID);
				current_node->pChildren.push_back(leaves[current_var][i]);
				current_node->weights.push_back(0.0);
				current_node->log_weights.push_back(0.0);
				current_node->log_inference.push_back(0.0);

				leaves[current_var][i]->parents.push_back(current_node->ID);
				leaves[current_var][i]->pParents.push_back(current_node);
			}

		} else if (current_node->Type == "Product") {
			int number_of_children = pow((double) current_node->scope.size(), (1/(double)current_node->Depth));
			//std::srand ( unsigned ( std::time(0) ) );
			std::random_shuffle( current_node->scope.begin(), current_node->scope.end() );
			int scope_size = current_node->scope.size() / number_of_children;

			for (int i=0; i < number_of_children; i++) {
				newNode = new spnNode;
				newNode->ID = current_id++;
				spnNetwork.push_back(newNode);
				search_queue.push(newNode);
				newNode->Type = "Sum";
				newNode->parents.push_back(current_node->ID);
				newNode->pParents.push_back(current_node);
				newNode->Depth = current_node->Depth - 1;

				for (int j = 0; j < scope_size; j++) {
					newNode->scope.push_back(current_node->scope[i*scope_size + j]);
				}
				if ( i == number_of_children -1 && (scope_size * number_of_children < current_node->scope.size())) {
					for (int j = scope_size * number_of_children; j < current_node->scope.size(); j++) {
						newNode->scope.push_back(current_node->scope[j]);
					}
				}
				current_node->children.push_back(newNode->ID);
				current_node->pChildren.push_back(newNode);
			}
			current_node->scope.clear();
		}
	}
	cout << "Number Of Nodes: " << spnNetwork.size() << endl;
}

int wsize(vector<int> &scope)
{
	return scope[3] - scope[2] + 1;
}
int hsize(vector<int> &scope)
{
	return scope[1] - scope[0] + 1;
}
void set_scope(vector<int> &scope, int minh, int maxh, int minw, int maxw)
{
	scope[0] = minh;
	scope[1] = maxh;
	scope[2] = minw;
	scope[3] = maxw;
}
void SPN::generateImageStructure(int height, int width, int min_scope_size, int step, int dont_merge_size, double branching_prob) {
	map<vector<int>, spnNode*> scope_map;
	
	vector<int> scope = { 0, height - 1, 0, width - 1 };
	int current_id = 0;
	spnNode* newNode;
	newNode = new spnNode;
	newNode->ID = current_id++;
	newNode->Depth = 0;
	root = 0;
	newNode->Type = "Sum";
		
	
	scope_map[scope] = newNode;
	newNode->iscope = scope;
	spnNetwork.push_back(newNode);
	queue<spnNode *> search_queue;
	search_queue.push(newNode);


	map<pair<int, int>, int> var_ind_map;
	int k = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			var_ind_map[make_pair(i, j)] = k++;
		}
	}
	// adding all leaf nodes
	for (int i = 0; i < numberOfVariables; i++) {
		for (int j = 0; j < numberOfValues; j++) {
			newNode = new spnNode;
			newNode->ID = current_id++;
			newNode->Type = "Var";
			newNode->featureIdx = i;
			newNode->ValueIdx = j;
			spnNetwork.push_back(newNode);
			leaves[i][j] = newNode;
		}
	}
	
	while (!search_queue.empty()) {
		spnNode *current_node = search_queue.front();
		search_queue.pop();

		// check if the size of the scope is less than min_scope_size, then expand
		scope = current_node->iscope;
		if (wsize(scope) > min_scope_size || hsize(scope) > min_scope_size)
		{  // decompose
			if (hsize(scope) > min_scope_size)
			{
				spnNode *up, *down;
				for (int i = scope[0]; i < scope[1]; i+=step)
				{
					if (((double)rand() / RAND_MAX) > (branching_prob * current_node->Depth) && (i != (scope[0] + scope[1])/2))
						continue;
					// decompose height
					newNode = new spnNode;
					spnNetwork.push_back(newNode);
					newNode->ID = current_id++;
					newNode->Type = "Product";
					newNode->iscope = current_node->iscope;
					newNode->parents.push_back(current_node->ID);
					newNode->pParents.push_back(current_node);
					current_node->children.push_back(newNode->ID);
					current_node->pChildren.push_back(newNode);
					current_node->weights.push_back(1);
					current_node->log_weights.push_back(0);
					// up
					vector<int> new_scope(4);
					set_scope(new_scope, scope[0], i, scope[2], scope[3]);

					if (scope_map.find(new_scope) != scope_map.end() && (hsize(new_scope) > dont_merge_size) && (wsize(new_scope) > dont_merge_size))
					{ // it exists
						up = scope_map[new_scope];
					}
					else
					{
						up = new spnNode;
						spnNetwork.push_back(up);
						up->ID = current_id++;
						up->Type = "Sum";
						up->iscope = new_scope;
						up->Depth = current_node->Depth + 1;
						scope_map[new_scope] = up;
						search_queue.push(up);
					}
					up->parents.push_back(newNode->ID);
					up->pParents.push_back(newNode);
					newNode->children.push_back(up->ID);
					newNode->pChildren.push_back(up);
					// down

					set_scope(new_scope, i + 1, scope[1], scope[2], scope[3]);
					if (scope_map.find(new_scope) != scope_map.end() && (hsize(new_scope) > dont_merge_size) && (wsize(new_scope) > dont_merge_size))
					{ // it exists
						down = scope_map[new_scope];
					}
					else
					{
						down = new spnNode;
						spnNetwork.push_back(down);
						down->ID = current_id++;
						down->Type = "Sum";
						down->iscope = new_scope;
						down->Depth = current_node->Depth + 1;
						scope_map[new_scope] = down;
						search_queue.push(down);
					}
					down->parents.push_back(newNode->ID);
					down->pParents.push_back(newNode);
					newNode->children.push_back(down->ID);
					newNode->pChildren.push_back(down);
				}
			}
			if (wsize(scope) > min_scope_size)
			{
				spnNode *up, *down;
				for (int i = scope[2]; i < scope[3]; i+=step)
				{
					if (((double)rand() / RAND_MAX) > (branching_prob * current_node->Depth) && (i != (scope[2] + scope[3]) / 2))
						continue;
					// decompose height
					newNode = new spnNode;
					spnNetwork.push_back(newNode);
					newNode->ID = current_id++;
					newNode->Type = "Product";
					newNode->iscope = current_node->iscope;
					newNode->parents.push_back(current_node->ID);
					newNode->pParents.push_back(current_node);
					current_node->children.push_back(newNode->ID);
					current_node->pChildren.push_back(newNode);
					current_node->weights.push_back(1);
					current_node->log_weights.push_back(0);
					// up
					vector<int> new_scope(4);
					set_scope(new_scope, scope[0], scope[1], scope[2], i);
					if (scope_map.find(new_scope) != scope_map.end() && (wsize(new_scope) > dont_merge_size ) && (wsize(new_scope) > dont_merge_size))
					{ // it exists
						up = scope_map[new_scope];

					}
					else
					{
						up = new spnNode;
						spnNetwork.push_back(up);
						up->ID = current_id++;
						up->Type = "Sum";
						up->iscope = new_scope;
						up->Depth = up->Depth + 1;
						scope_map[new_scope] = up;
						search_queue.push(up);
					}
					up->parents.push_back(newNode->ID);
					up->pParents.push_back(newNode);
					newNode->children.push_back(up->ID);
					newNode->pChildren.push_back(up);
					// down

					set_scope(new_scope, scope[0], scope[1], i + 1, scope[3]);
					if (scope_map.find(new_scope) != scope_map.end() && (wsize(new_scope) > dont_merge_size ) && (wsize(new_scope) > dont_merge_size))
					{ // it exists
						down = scope_map[new_scope];
					}
					else
					{
						down = new spnNode;
						spnNetwork.push_back(down);
						down->ID = current_id++;
						down->Type = "Sum";
						down->iscope = new_scope;
						down->Depth = current_node->Depth + 1;
						scope_map[new_scope] = down;
						search_queue.push(down);
					}
					down->parents.push_back(newNode->ID);
					down->pParents.push_back(newNode);
					newNode->children.push_back(down->ID);
					newNode->pChildren.push_back(down);
				}
			}
		}
		else
		{ // generate a joint leaf dist
			map<int, pair<int, int>> var_map;
			int k = 0;
			for (int i = scope[0]; i <= scope[1]; i++)
			{
				for (int j = scope[2]; j <= scope[3]; j++)
				{
					var_map[k++] = make_pair(i, j);
				}
			}
			vector<vector<int>> truth_table(k, vector<int>(pow(2, k)));
			for (int level = 0; level < k; level++) {
				int ind = 0;
				for (int i = (1 << k) - 1; i >= 0; i--)   // we'll always output 2**n bits
					truth_table[level][ind++] = (i >> level) & 1;
			};
			for (int i = 0; i < pow(2, k); i++)
			{
				if (k > 1)
				{
					newNode = new spnNode;
					spnNetwork.push_back(newNode);
					newNode->ID = current_id++;
					newNode->Type = "Product";
					newNode->iscope = current_node->iscope;
					newNode->parents.push_back(current_node->ID);
					newNode->pParents.push_back(current_node);
					current_node->children.push_back(newNode->ID);
					current_node->pChildren.push_back(newNode);
				}
				else
					newNode = current_node;

				current_node->weights.push_back(1);
				current_node->log_weights.push_back(0);

				for (int j = 0; j < k; j++)
				{
					newNode->children.push_back(leaves[var_ind_map[var_map[j]]][truth_table[j][i]]->ID);
					newNode->pChildren.push_back(leaves[var_ind_map[var_map[j]]][truth_table[j][i]]);
					leaves[var_ind_map[var_map[j]]][truth_table[j][i]]->parents.push_back(newNode->ID);
					leaves[var_ind_map[var_map[j]]][truth_table[j][i]]->pParents.push_back(newNode);
				}
			}
		}
	}
	
	cout << "Number Of Nodes: " << spnNetwork.size() << endl;
}





void SPN::generateContinuousRandomStructure(int Depth, int sum_max_num_children) {
	int current_id = 0;
	spnNode* newNode;
	newNode = new spnNode;
	newNode->ID = current_id++;
	root = 0;
	newNode->Type = "Sum";
	newNode->Depth = Depth;
	newNode->mixture_node = false;
	for (int i = 0; i < numberOfVariables; i++) {
		newNode->scope.push_back(i);
	}
	spnNetwork.push_back(newNode);
	queue<spnNode *> search_queue;
	search_queue.push(newNode);

	// adding all leaf nodes
	for (int i = 0; i < numberOfVariables; i++) {
		for (int j = 0; j < numberOfValues; j++) {
			newNode = new spnNode;
			newNode->ID = current_id++;
			newNode->Type = "Var";
			newNode->featureIdx = i;
			newNode->ValueIdx = j;
			newNode->mixture_node = false;
			spnNetwork.push_back(newNode);
			leaves[i][j] = newNode;
		}
	}
	while (!search_queue.empty()) {
		spnNode *current_node = search_queue.front();
		search_queue.pop();
		if (current_node->Type == "Sum" && current_node->scope.size() > 1) {
			//int number_of_children = std::min(pow((double)current_node->scope.size(), (1 - 1 / (double)current_node->Depth)), (double)sum_max_num_children);
			int number_of_children = sum_max_num_children;

			//int number_of_children = std::max(pow((double)current_node->scope.size(), (1 - 1 / (double)current_node->Depth)), (double)sum_max_num_children);
			for (int i = 0; i < number_of_children; i++) {
				newNode = new spnNode;
				newNode->ID = current_id++;
				spnNetwork.push_back(newNode);
				search_queue.push(newNode);

				newNode->Type = "Product";
				newNode->parents.push_back(current_node->ID);
				newNode->pParents.push_back(current_node);
				newNode->Depth = current_node->Depth;
				newNode->scope = current_node->scope;
				newNode->mixture_node = false;

				current_node->children.push_back(newNode->ID);
				current_node->pChildren.push_back(newNode);
				current_node->weights.push_back(0.0);
				current_node->log_weights.push_back(0.0);
				current_node->log_inference.push_back(0.0);
			}
			current_node->scope.clear();
		}
		else if (current_node->Type == "Sum" && current_node->scope.size() == 1) {
			current_node->mixture_node = true;
			current_node->gaussian_mixture.resize(numberOfMixtures);
			current_node->updated_gaussian_mixture.resize(numberOfMixtures);
			current_node->updated_mixture_constants.resize(numberOfMixtures);
			int current_var = current_node->scope[0];
			for (int i = 0; i < numberOfValues; i++) {
				current_node->children.push_back(leaves[current_var][i]->ID);
				current_node->pChildren.push_back(leaves[current_var][i]);
				current_node->weights.push_back(0.0);
				current_node->log_weights.push_back(0.0);
				current_node->log_inference.push_back(0.0);
				current_node->featureIdx = current_var;


				leaves[current_var][i]->parents.push_back(current_node->ID);
				leaves[current_var][i]->pParents.push_back(current_node);
			}

		}
		else if (current_node->Type == "Product") {
			int number_of_children = pow((double)current_node->scope.size(), (1 / max(((double)current_node->Depth)-1,1.0)));
			//int number_of_children = current_node->scope.size();
			//std::srand(unsigned(std::time(0)));
			std::random_shuffle(current_node->scope.begin(), current_node->scope.end());
			int scope_size = current_node->scope.size() / number_of_children;

			for (int i = 0; i < number_of_children; i++) {
				newNode = new spnNode;
				newNode->ID = current_id++;
				spnNetwork.push_back(newNode);
				search_queue.push(newNode);
				newNode->Type = "Sum";
				newNode->parents.push_back(current_node->ID);
				newNode->pParents.push_back(current_node);
				newNode->Depth = current_node->Depth - 1;
				newNode->mixture_node = false;

				for (int j = 0; j < scope_size; j++) {
					newNode->scope.push_back(current_node->scope[i*scope_size + j]);
				}
				if (i == number_of_children - 1 && (scope_size * number_of_children < current_node->scope.size())) {
					for (int j = scope_size * number_of_children; j < current_node->scope.size(); j++) {
						newNode->scope.push_back(current_node->scope[j]);
					}
				}
				current_node->children.push_back(newNode->ID);
				current_node->pChildren.push_back(newNode);
			}
			current_node->scope.clear();
		}
	}
	cout << "Number Of Nodes: " << spnNetwork.size() << endl;
}

void SPN::generateContinuousRandomStructure(int Depth, int sum_max_num_children, int leaf_n_variables) {
	int current_id = 0;
	spnNode* newNode;
	newNode = new spnNode;
	newNode->ID = current_id++;
	root = 0;
	newNode->Type = "Sum";
	newNode->Depth = Depth;
	newNode->mixture_node = false;
	for (int i = 0; i < numberOfVariables; i++) {
		newNode->scope.push_back(i);
	}
	spnNetwork.push_back(newNode);
	queue<spnNode *> search_queue;
	search_queue.push(newNode);

	// adding all leaf nodes
	for (int i = 0; i < numberOfVariables; i++) {
		for (int j = 0; j < numberOfValues; j++) {
			newNode = new spnNode;
			newNode->ID = current_id++;
			newNode->Type = "Var";
			newNode->featureIdx = i;
			newNode->ValueIdx = j;
			newNode->mixture_node = false;
			spnNetwork.push_back(newNode);
			leaves[i][j] = newNode;
		}
	}
	while (!search_queue.empty()) {
		spnNode *current_node = search_queue.front();
		search_queue.pop();
		if (current_node->Type == "Sum" && current_node->scope.size() > 1) {
			//int number_of_children = std::min(pow((double)current_node->scope.size(), (1 - 1 / (double)current_node->Depth)), (double)sum_max_num_children);
			int number_of_children = sum_max_num_children;

			//int number_of_children = std::max(pow((double)current_node->scope.size(), (1 - 1 / (double)current_node->Depth)), (double)sum_max_num_children);
			for (int i = 0; i < number_of_children; i++) {
				newNode = new spnNode;
				newNode->ID = current_id++;
				spnNetwork.push_back(newNode);
				search_queue.push(newNode);

				newNode->Type = "Product";
				newNode->parents.push_back(current_node->ID);
				newNode->pParents.push_back(current_node);
				newNode->Depth = current_node->Depth;
				newNode->scope = current_node->scope;
				newNode->mixture_node = false;

				current_node->children.push_back(newNode->ID);
				current_node->pChildren.push_back(newNode);
				current_node->weights.push_back(0.0);
				current_node->log_weights.push_back(0.0);
				current_node->log_inference.push_back(0.0);
			}
			current_node->scope.clear();
		}
		else if (current_node->Type == "Sum" && current_node->scope.size() == 1) {
			current_node->mixture_node = true;
			current_node->gaussian_mixture.resize(numberOfMixtures);
			current_node->updated_gaussian_mixture.resize(numberOfMixtures);
			current_node->updated_mixture_constants.resize(numberOfMixtures);
			int current_var = current_node->scope[0];
			for (int i = 0; i < numberOfValues; i++) {
				current_node->children.push_back(leaves[current_var][i]->ID);
				current_node->pChildren.push_back(leaves[current_var][i]);
				current_node->weights.push_back(0.0);
				current_node->log_weights.push_back(0.0);
				current_node->log_inference.push_back(0.0);
				current_node->featureIdx = current_var;


				leaves[current_var][i]->parents.push_back(current_node->ID);
				leaves[current_var][i]->pParents.push_back(current_node);
			}

		}
		else if (current_node->Type == "Product") {
			int number_of_children = pow((double)current_node->scope.size(), (1 / max(((double)current_node->Depth) - 1, 1.0)));
			//int number_of_children = current_node->scope.size();
			//std::srand(unsigned(std::time(0)));
			std::random_shuffle(current_node->scope.begin(), current_node->scope.end());
			int scope_size = current_node->scope.size() / number_of_children;

			for (int i = 0; i < number_of_children; i++) {
				newNode = new spnNode;
				newNode->ID = current_id++;
				spnNetwork.push_back(newNode);
				search_queue.push(newNode);
				newNode->Type = "Sum";
				newNode->parents.push_back(current_node->ID);
				newNode->pParents.push_back(current_node);
				newNode->Depth = current_node->Depth - 1;
				newNode->mixture_node = false;

				for (int j = 0; j < scope_size; j++) {
					newNode->scope.push_back(current_node->scope[i*scope_size + j]);
				}
				if (i == number_of_children - 1 && (scope_size * number_of_children < current_node->scope.size())) {
					for (int j = scope_size * number_of_children; j < current_node->scope.size(); j++) {
						newNode->scope.push_back(current_node->scope[j]);
					}
				}
				current_node->children.push_back(newNode->ID);
				current_node->pChildren.push_back(newNode);
			}
			current_node->scope.clear();
		}
	}
	cout << "Number Of Nodes: " << spnNetwork.size() << endl;
}
void SPN::generateContinuousMixtureMultivariate(int n_mixtures) {
	int Depth = 1;
	int sum_max_num_children;
	int leaf_n_variables = numberOfVariables;


	int current_id = 0;
	spnNode* newNode;
	newNode = new spnNode;
	newNode->ID = current_id++;
	root = 0;
	newNode->Type = "Sum";
	newNode->Depth = Depth;
	newNode->mixture_node = false;
	for (int i = 0; i < numberOfVariables; i++) {
		newNode->scope.push_back(i);
	}
	spnNetwork.push_back(newNode);
	queue<spnNode *> search_queue;
	search_queue.push(newNode);

	// adding all leaf nodes
	for (int i = 0; i < numberOfVariables; i++) {
		for (int j = 0; j < numberOfValues; j++) {
			newNode = new spnNode;
			newNode->ID = current_id++;
			newNode->Type = "Var";
			newNode->featureIdx = i;
			newNode->ValueIdx = j;
			newNode->mixture_node = false;
			spnNetwork.push_back(newNode);
			leaves[i][j] = newNode;
		}
	}
	int mixture_sum_node = false;
	while (!search_queue.empty()) {
		spnNode *current_node = search_queue.front();
		search_queue.pop();
		if (current_node->Type == "Sum" && current_node->scope.size() >= leaf_n_variables && !mixture_sum_node) {
			mixture_sum_node = true;
			//int number_of_children = std::min(pow((double)current_node->scope.size(), (1 - 1 / (double)current_node->Depth)), (double)sum_max_num_children);
			int number_of_children = n_mixtures;

			//int number_of_children = std::max(pow((double)current_node->scope.size(), (1 - 1 / (double)current_node->Depth)), (double)sum_max_num_children);
			for (int i = 0; i < number_of_children; i++) {
				newNode = new spnNode;
				newNode->ID = current_id++;
				spnNetwork.push_back(newNode);
				search_queue.push(newNode);

				newNode->Type = "Product";
				newNode->parents.push_back(current_node->ID);
				newNode->pParents.push_back(current_node);
				newNode->Depth = current_node->Depth;
				newNode->scope = current_node->scope;
				newNode->mixture_node = false;

				current_node->children.push_back(newNode->ID);
				current_node->pChildren.push_back(newNode);
				current_node->weights.push_back(0.0);
				current_node->log_weights.push_back(0.0);
				current_node->log_inference.push_back(0.0);
			}
			current_node->scope.clear();
		}
		else if (current_node->Type == "Sum" && current_node->scope.size() == leaf_n_variables && mixture_sum_node) {
			current_node->mixture_node = true;
			//current_node->gaussian_mixture.resize(numberOfMixtures);
			//current_node->updated_gaussian_mixture.resize(numberOfMixtures);
			//current_node->updated_mixture_constants.resize(numberOfMixtures);
			for (int j = 0; j < current_node->scope.size(); j++) {
				int current_var = current_node->scope[j];
				current_node->children.push_back(leaves[current_var][0]->ID);
				current_node->pChildren.push_back(leaves[current_var][0]);
				//current_node->weights.push_back(0.0);
				//current_node->log_weights.push_back(0.0);
				//current_node->log_inference.push_back(0.0);
				//current_node->featureIdx = current_var;
				leaves[current_var][0]->parents.push_back(current_node->ID);
				leaves[current_var][0]->pParents.push_back(current_node);
			}
		}
		else if (current_node->Type == "Product") {
			int number_of_children = 1;
			//int number_of_children = current_node->scope.size();
			//std::srand(unsigned(std::time(0)));
			//std::random_shuffle(current_node->scope.begin(), current_node->scope.end());
			int scope_size = current_node->scope.size() / number_of_children;

			for (int i = 0; i < number_of_children; i++) {
				newNode = new spnNode;
				newNode->ID = current_id++;
				spnNetwork.push_back(newNode);
				search_queue.push(newNode);
				newNode->Type = "Sum";
				newNode->parents.push_back(current_node->ID);
				newNode->pParents.push_back(current_node);
				newNode->Depth = current_node->Depth - 1;
				newNode->mixture_node = false;

				for (int j = 0; j < scope_size; j++) {
					newNode->scope.push_back(current_node->scope[i*scope_size + j]);
				}
				if (i == number_of_children - 1 && (scope_size * number_of_children < current_node->scope.size())) {
					for (int j = scope_size * number_of_children; j < current_node->scope.size(); j++) {
						newNode->scope.push_back(current_node->scope[j]);
					}
				}
				current_node->children.push_back(newNode->ID);
				current_node->pChildren.push_back(newNode);
			}
			current_node->scope.clear();
		}
	}
	cout << "Number Of Nodes: " << spnNetwork.size() << endl;
}

void SPN::generateContinuousRandomStructureMultivariate(int Depth, int sum_max_num_children, int leaf_n_variables) {
	int current_id = 0;
	spnNode* newNode;
	newNode = new spnNode;
	newNode->ID = current_id++;
	root = 0;
	newNode->Type = "Sum";
	newNode->Depth = Depth;
	newNode->mixture_node = false;
	for (int i = 0; i < numberOfVariables; i++) {
		newNode->scope.push_back(i);
	}
	spnNetwork.push_back(newNode);
	queue<spnNode *> search_queue;
	search_queue.push(newNode);

	// adding all leaf nodes
	for (int i = 0; i < numberOfVariables; i++) {
		for (int j = 0; j < numberOfValues; j++) {
			newNode = new spnNode;
			newNode->ID = current_id++;
			newNode->Type = "Var";
			newNode->featureIdx = i;
			newNode->ValueIdx = j;
			newNode->mixture_node = false;
			spnNetwork.push_back(newNode);
			leaves[i][j] = newNode;
		}
	}
	while (!search_queue.empty()) {
		spnNode *current_node = search_queue.front();
		search_queue.pop();
		if (current_node->Type == "Sum" && current_node->scope.size() > leaf_n_variables) {
			//int number_of_children = std::min(pow((double)current_node->scope.size(), (1 - 1 / (double)current_node->Depth)), (double)sum_max_num_children);
			int number_of_children = sum_max_num_children;

			//int number_of_children = std::max(pow((double)current_node->scope.size(), (1 - 1 / (double)current_node->Depth)), (double)sum_max_num_children);
			for (int i = 0; i < number_of_children; i++) {
				newNode = new spnNode;
				newNode->ID = current_id++;
				spnNetwork.push_back(newNode);
				search_queue.push(newNode);

				newNode->Type = "Product";
				newNode->parents.push_back(current_node->ID);
				newNode->pParents.push_back(current_node);
				newNode->Depth = current_node->Depth;
				newNode->scope = current_node->scope;
				newNode->mixture_node = false;

				current_node->children.push_back(newNode->ID);
				current_node->pChildren.push_back(newNode);
				current_node->weights.push_back(0.0);
				current_node->log_weights.push_back(0.0);
				current_node->log_inference.push_back(0.0);
			}
			current_node->scope.clear();
		}
		else if (current_node->Type == "Sum" && current_node->scope.size() <= leaf_n_variables) {
			current_node->mixture_node = true;
			//current_node->gaussian_mixture.resize(numberOfMixtures);
			//current_node->updated_gaussian_mixture.resize(numberOfMixtures);
			//current_node->updated_mixture_constants.resize(numberOfMixtures);
			for (int j = 0; j < current_node->scope.size(); j++) {
				int current_var = current_node->scope[j];
				current_node->children.push_back(leaves[current_var][0]->ID);
				current_node->pChildren.push_back(leaves[current_var][0]);
				//current_node->weights.push_back(0.0);
				//current_node->log_weights.push_back(0.0);
				//current_node->log_inference.push_back(0.0);
				//current_node->featureIdx = current_var;
				leaves[current_var][0]->parents.push_back(current_node->ID);
				leaves[current_var][0]->pParents.push_back(current_node);
			}

		}
		else if (current_node->Type == "Product") {
			int number_of_children = 2; // pow((double)current_node->scope.size(), (1 / max(((double)current_node->Depth) - 1, 1.0)));
			//int number_of_children = current_node->scope.size();
			//std::srand(unsigned(std::time(0)));
			std::random_shuffle(current_node->scope.begin(), current_node->scope.end());
			int scope_size = current_node->scope.size() / number_of_children;

			for (int i = 0; i < number_of_children; i++) {
				newNode = new spnNode;
				newNode->ID = current_id++;
				spnNetwork.push_back(newNode);
				search_queue.push(newNode);
				newNode->Type = "Sum";
				newNode->parents.push_back(current_node->ID);
				newNode->pParents.push_back(current_node);
				newNode->Depth = current_node->Depth - 1;
				newNode->mixture_node = false;

				for (int j = 0; j < scope_size; j++) {
					newNode->scope.push_back(current_node->scope[i*scope_size + j]);
				}
				if (i == number_of_children - 1 && (scope_size * number_of_children < current_node->scope.size())) {
					for (int j = scope_size * number_of_children; j < current_node->scope.size(); j++) {
						newNode->scope.push_back(current_node->scope[j]);
					}
				}
				current_node->children.push_back(newNode->ID);
				current_node->pChildren.push_back(newNode);
			}
			current_node->scope.clear();
		}
	}
	cout << "Number Of Nodes: " << spnNetwork.size() << endl;
}
SPN::~SPN(void)
{
}

