/*
 * SPN.h
 *
 *  Created on: Jun 20, 2015
 *      Author: abdoo_000
 */

#ifndef SPN_H_
#define SPN_H_

#include <vector>
#include <string>
#include <Eigen/Dense>

const int SIZE_OF_LEARNING_VECTOR = 10;
struct mixture {
	double NG[4];
	double Normal[2];
};
struct multivariate_mixture {
	double k;
	Eigen::MatrixXd mu;
	Eigen::MatrixXd W;
	double v; // v > d + 1
};
class image_scope
{
public:
	int min_height;
	int max_height;
	int min_width;
	int max_width;
	int wsize()
	{
		return max_width - min_width + 1;
	}
	int hsize()
	{
		return max_height - min_height + 1;
	}
	void set(int minh, int maxh, int minw, int maxw)
	{
		min_height = minh;
		max_height = maxh;
		min_width = minw;
		max_width = maxw;
	}
	bool operator<(const image_scope & n) const {
		return (this->min_width < n.min_width ||
			(this->min_width == n.min_width && this->min_height < n.min_height) ||
			(this->min_height == n.min_height && this->max_height < n.max_height) ||
			(this->max_height == n.max_height && this->max_width < n.max_width) );   // for example
	}
};
struct spnNode{
	int ID;
	std::string Type;
	std::vector<int> children;
	std::vector<spnNode*> pChildren; //pointer to children
	std::vector<int> parents;
	std::vector<spnNode*> pParents;
	std::vector<double> weights;
	std::vector<double> log_weights;
	std::vector<double> log_inference;
	std::vector<int> scope;
	std::vector<int> iscope; // image score , 4 number, minh, maxh, minw, maxw
	int DistType; // 0 == discrete, 1 == Gaussian
	std::vector<mixture> gaussian_mixture;
	std::vector<mixture> updated_gaussian_mixture;
	multivariate_mixture mmixture;
	multivariate_mixture updated_mmixture;
	double mmixture_constant;
	std::vector<double> updated_mixture_constants;
	bool mixture_node;
	int Depth;


	double learning_vector[SIZE_OF_LEARNING_VECTOR];
	int featureIdx;
	std::vector<int> leaf_scope;

	int ValueIdx;
	double inferenceValue;
	double log_inferenceValue;
	spnNode* next;
};

class SPN {
public:
	SPN(int numVariables, int numValues);

	int readSPN(std::string filePath);
	int readSPNContinuousHan(std::string filePath);
	int readtrainData(std::string filePath);
	int readtestData(std::string filePath);
	void writeDataCont(std::string filePath, std::vector<double*> data);
	void writeSPNContinuous(std::string file_name);
	void writeSPNContinuousMultivariate(std::string file_name);
	int readtestDataCont(std::string filePath);
	int readtrainDataCont(std::string filePath);
	void convertUciDataset(std::string ifile, std::string ofile);
	void generateTrainTestUciDatasets(std::string ifile, std::string trainfile, std::string testfile, int number_of_partitions);
	double doInference(std::vector<int*> & data, std::string file_path);
	double doInferenceContinuous(std::vector<double*> &data, std::string file_path);
	double doInferenceContinuousMultivariate(std::vector<double*> &data, std::string file_path);
	double doInferenceOnUCI(std::string ifile);
	double doInferenceOnInstance(int* Instance);
	double doInferenceOnInstanceContinuous(double* Instance);
	double doInferenceOnInstanceContinuousMultivariate(double* Instance);
	void doMomentMatching(std::vector<int*> data);
	void doMomentMatchingOnUCI(std::string ifile);
	void doMomentMatchingOnInstance(int* Instance);
	void doMomentMatchingOnInstanceContinuous(double* Instance);
	void doMomentMatchingOnInstanceContinuousMultivariate(double* Instance);
	void doMomentMatchingOnInstanceContinuous1(double* Instance);
	void doUpwardPass(int *Instance);
	void doDownwardPass(int *Instance);
	void updateWeights();
	void doUpwardPassContinuous(double *Instance);
	void doUpwardPassContinuousMultivariate(double *Instance);
	void doUpwardPassContinuous1(double *Instance);
	void doDownwardPassContinuous(double *Instance);
	//void doDownwardPassContinuousMultivariate(double *Instance);
	void doMomentMatchingContinuous(std::vector<double*> data);
	void doMomentMatchingContinuousMultivariate(std::vector<double*> data);
	void updateWeightsContinuous();
	void updateWeightsContinuousMultivariate();
	void updateWeightsContinuous1();
	void normalizeWeights(double smoothing_factor = 0);
	void randomizeWeights();
	void randomizeWeightsContinuous();
	void randomizeWeightsContinuousMultivariate();
	void writeSPN(std::string file_name);
	void writeSPNMyFormat(std::string file_name);
	void writeSPNMyFormatContinuous(std::string file_name);
	int readSPNContinuous(std::string filePath);
	void generateRandomStructure(int Depth, int sum_max_num_children);
	void generateContinuousRandomStructure(int Depth, int sum_max_num_children);
	void generateImageStructure(int height, int width, int min_scope_size, int step, int dont_merge_size, double branching_prob);
	void generateContinuousRandomStructure(int Depth, int sum_max_num_children, int leaf_n_variables);
	void generateContinuousMixtureMultivariate(int n_mixtures);
	//void generateContinuousRandomStructureMultivariate(int n_mixtures);
	void generateContinuousRandomStructureMultivariate(int Depth, int sum_max_num_children, int leaf_n_variables);
	void printMixtures();
	double* sampleInstanceContinuous();
	std::vector<double*> generateData(int data_size);
	std::vector<int*> testData;
	std::vector<int*> trainData;
    std::vector<double*> trainDataCont;
	std::vector<double*> testDataCont;
	~SPN(void);
	std::vector<spnNode*> spnNetwork;
	

private:
	double scale;
	int numberOfVariables;
	int numberOfValues;
	int numberOfInstances;
	int numberOfMixtures;


	std::vector<std::vector<spnNode*> > leaves;


	int root, current;
	
};

#endif /* SPN_H_ */
