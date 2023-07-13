//
// Created by Khurram Javed on 2022-02-23.
//

#ifndef INCLUDE_NN_NETWORKS_LSTM_BPTT_H_
#define INCLUDE_NN_NETWORKS_LSTM_BPTT_H_

#include <vector>
#include <queue>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../lstm.h"
#include "../dynamic_elem.h"
#include "./neural_network.h"
#include "base_lstm.h"

class DenseLSTM : public BaseLSTM{
 protected:
  int64_t time_step;
  std::mt19937 mt;

 public:
//
  float step_size;
  std::vector<float> get_state();
  int truncation;
  int input_size;
  int hidden_state_size;
  std::vector<float> prediction_weights;
  std::vector<float> prediction_weights_grad;

  void print_features_stats() override;

  std::vector<std::vector<float>> x_queue;
  std::vector<std::vector<float>> h_queue;
  std::vector<std::vector<float>> c_queue;
  std::vector<std::vector<float>> i_queue;
  std::vector<std::vector<float>> g_queue;
  std::vector<std::vector<float>> f_queue;
  std::vector<std::vector<float>> o_queue;

  std::vector<float> W;
  std::vector<float> W_grad;
  std::vector<float> U;
  std::vector<float> U_grad;
  std::vector<float> b;
  std::vector<float> b_grad;

  float  get_target_without_sideeffects(std::vector<float> inputs) override;

  std::vector<LinearNeuron> input_neurons;

  std::vector<LSTM> LSTM_neurons;

  std::vector<float> read_output_values();

  DenseLSTM(float step_size,
            int seed,
            int hidden_size,
            int no_of_input_features,
            int truncation);

  void zero_grad() override;

  void decay_gradient(float decay_rate) override;

  float forward(std::vector<float> inputs) override;

  void backward(int layer) override;

  std::vector<std::vector<float>> backward_with_future_grad(std::vector<std::vector<float>> grad_f, int time);

  void virtual update_parameters(int layer, float error) override;

  void reset_state();
  std::vector<float> get_normalized_state();

};

class DenseLSTMRmsProp : public  DenseLSTM{
  std::vector<float> W_grad_rmsprop;
  std::vector<float> U_grad_rmsprop;
  std::vector<float> b_grad_rmsprop;
  std::vector<float> prediction_weights_grad_rmsprop;
  float beta_2;
  float epsilon;
 public:
  DenseLSTMRmsProp(float step_size,
                   int seed,
                   int hidden_size,
                   int no_of_input_features,
                   int truncation, float beta_2, float epsilon);
  void update_parameters(int layer, float error) override;

};

#endif //INCLUDE_NN_NETWORKS_LSTM_BPTT_H_
