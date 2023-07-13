//
// Created by Khurram Javed on 2022-07-28.
//

#include "../../../include/nn/networks/network_factory.h"
#include "../../../include/experiment/Experiment.h"
#include "../../../include/nn/networks/lstm_bptt.h"
#include "../../../include/nn/networks/lstm_incremental_networks.h"
#include <string>
#include <vector>

BaseLSTM *NetworkFactory::get_network(Experiment *experiment_config) {

  BaseLSTM *network;
  std::cout << "Layer size = " << experiment_config->get_vector_param("features")[1] << std::endl;
  if (experiment_config->get_string_param("algorithm") == "constructive") {
    network = new IncrementalNetworks(
        experiment_config->get_float_param("step_size"),
        experiment_config->get_int_param("seed"),
        experiment_config->get_int_param("input_features"), 1,
        std::stoi(experiment_config->get_vector_param("features")[0]), 1,
        experiment_config->get_float_param("std_cap"),
        experiment_config->get_float_param("normalization_rate"));
  } else if (experiment_config->get_string_param("algorithm") == "columnar") {
    network = new IncrementalNetworks(
        experiment_config->get_float_param("step_size"),
        experiment_config->get_int_param("seed"),
        experiment_config->get_int_param("input_features"), 1,
        std::stoi(experiment_config->get_vector_param("features")[0]),
        experiment_config->get_int_param("features"),
        experiment_config->get_float_param("std_cap"),
        experiment_config->get_float_param("normalization_rate"));
  } else if (experiment_config->get_string_param("algorithm") == "hybrid") {
    network = new IncrementalNetworks(
        experiment_config->get_float_param("step_size"),
        experiment_config->get_int_param("seed"),
        experiment_config->get_int_param("input_features"), 1,
        std::stoi(experiment_config->get_vector_param("features")[0]),
        std::stoi(experiment_config->get_vector_param("features")[1]),
        experiment_config->get_float_param("std_cap"),
        experiment_config->get_float_param("normalization_rate"));
  } else if (experiment_config->get_string_param("algorithm") ==
             "hybrid_normalized") {
    network = new NormalizedIncrementalNetworks(
        experiment_config->get_float_param("step_size"),
        experiment_config->get_int_param("seed"),
        experiment_config->get_int_param("input_features"), 1,
        std::stoi(experiment_config->get_vector_param("features")[0]),
        std::stoi(experiment_config->get_vector_param("features")[1]),
        experiment_config->get_float_param("std_cap"),
        experiment_config->get_float_param("normalization_rate"));
  } else if (experiment_config->get_string_param("algorithm") == "snap") {
    network = new Snap1(experiment_config->get_float_param("step_size"),
                        experiment_config->get_int_param("seed"),
                        experiment_config->get_int_param("input_features"), 1,
                        std::stoi(experiment_config->get_vector_param("features")[0]),
                        std::stoi(experiment_config->get_vector_param("features")[1]),
                        experiment_config->get_float_param("std_cap"));
  } else if (experiment_config->get_string_param("algorithm") == "tbptt") {
    network = new DenseLSTM(experiment_config->get_float_param("step_size"),
                            experiment_config->get_int_param("seed"),
                            std::stoi(experiment_config->get_vector_param(
                                "features_truncation")[1]),
                            experiment_config->get_int_param("input_features"),
                            std::stoi(experiment_config->get_vector_param(
                                "features_truncation")[0]));
  } else if (experiment_config->get_string_param("algorithm") ==
             "tbptt_rmsprop") {
    network = new DenseLSTMRmsProp(
        experiment_config->get_float_param("step_size"),
        experiment_config->get_int_param("seed"),
        std::stoi(
            experiment_config->get_vector_param("features_truncation")[1]),
        experiment_config->get_int_param("input_features"),
        std::stoi(
            experiment_config->get_vector_param("features_truncation")[0]),
        experiment_config->get_float_param("beta_2"),
        experiment_config->get_float_param("eps"));
  }

  return network;
}