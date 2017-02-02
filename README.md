# BMM
This is an implementation for Bayesian Moment Matching algorithm for learning the parameters for Sum Product Networks (SPNs) with Discrete or Continuous variables.

To build this project, you need to install Eigen library. You can run the code using one of the following commands depending on the type of data.

Discrete data:
RandomStructureAndMM <number of variables> <depth of spn> <max number of children per sum node> <train data> <test data>

Continuous data using SPN model that is equivalent to GMMs:
BMM Multivariate_GMM <number_of_variables> <num_of_mixtures> <path_to_train> <path_to_test>

Continuous data using SPN model with random strucature:
BMM Multivariate_SPN <number_of_variables> <depth_of_spn> <max_number_of_children_for_sum_nodes> <number_of_variables_at_leaf_nodes> <path_to_train> <path_to_test>
