//
// Created by Khurram Javed on 2022-01-24.
//

#ifndef INCLUDE_NN_NETWORKS_LSTM_INCREMENTAL_NETWORKS_H_
#define INCLUDE_NN_NETWORKS_LSTM_INCREMENTAL_NETWORKS_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"
#include "./neural_network.h"
#include "../lstm.h"
#include "base_lstm.h"

class IncrementalNetworks : public BaseLSTM {
 protected:
  int64_t time_step;
  std::mt19937 mt;

 public:
//
//

  float step_size;
  float predictions;
  float bias;
  float bias_gradients;
  float bias_beta_2;
  float layer_size;
  float std_cap;
  float decay_rate;

  std::vector<int> indexes;
  std::vector<int> indexes_lstm_cells;

  std::vector<float> prediction_weights;

  std::vector<float> feature_mean;

  std::vector<float> feature_std;

  std::vector<float> avg_feature_value;

  std::vector<float> prediction_weights_gradient;

  std::vector<float> prediction_weights_beta_2;

  float  get_target_without_sideeffects(std::vector<float> inputs) override;

  std::vector<LinearNeuron> input_neurons;

  std::vector<LSTM*> LSTM_neurons;

  void print_features_stats() override;

  std::vector<float> real_all_running_mean();

  std::vector<float> read_all_running_variance();

  float read_output_values();

  IncrementalNetworks();

  IncrementalNetworks(float step_size, int seed, int no_of_input_features, int total_targets, int total_recurrent_features, int layer_size, float std_cap, float decay_rate);

  ~IncrementalNetworks();

  float forward(std::vector<float> inputs) override;

  void zero_grad() override;

  void decay_gradient(float decay_rate) override;

  virtual void backward(int layer) override;

  virtual void update_parameters(int layer, float error) override;

  void update_parameters_no_freeze(float error);

  std::vector<float> get_prediction_gradients();
  std::vector<float> get_prediction_weights();

  std::vector<float> get_state();
  std::vector<float> get_normalized_state();

  void reset_state();

};

class Snap1 : public IncrementalNetworks{
 public:
  Snap1(float step_size, int seed, int no_of_input_features, int total_targets, int total_recurrent_features, int layer_size, float std_cap);

  ~Snap1();

  void backward(int layer) override;

  void update_parameters(int layer, float error) override;


};

class NormalizedIncrementalNetworks : public IncrementalNetworks{
public:
  NormalizedIncrementalNetworks(float step_size, int seed, int no_of_input_features, int total_targets, int total_recurrent_features, int layer_size, float std_cap, float decay_rate);
};



#endif //INCLUDE_NN_NETWORKS_LSTM_INCREMENTAL_NETWORKS_H_
