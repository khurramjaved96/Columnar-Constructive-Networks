//
// Created by Khurram Javed on 2021-09-20.
//

#ifndef INCLUDE_NN_SYNCED_NEURON_H_
#define INCLUDE_NN_SYNCED_NEURON_H_


#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "./dynamic_elem.h"
#include "./synapse.h"
#include "./utils.h"


class Neuron : public dynamic_elem {
 public:
  bool frozen;
  bool is_recurrent_neuron;
  static int64_t neuron_id_generator;
  static std::mt19937 gen;
  bool is_input_neuron;
  bool is_bias_unit;
  int layer_number;
  float value;
  float pre_sync_value;
  int drinking_age;
  float value_before_firing;
  float neuron_utility;
  float neuron_utility_to_distribute;
  float sum_of_utility_traces;
  float running_mean;
  float running_variance;
  bool is_output_neuron;
  bool useless_neuron;
  int64_t id;
  float neuron_utility_trace_decay_rate;
  int neuron_age;

  void set_layer_number(int layer);

  int get_layer_number();

  void forward_gradients();

  virtual void update_value();


  std::vector<Synapse *> outgoing_synapses;
  std::vector<Synapse *> incoming_synapses;

  Neuron(bool is_input, bool is_output);

  virtual void fire();

  float introduce_targets(float target);

  virtual float backward(float output_grad) = 0;

  virtual float forward(float temp_value) = 0;

  void update_utility();

  float get_utility();

  bool is_mature();

  ~Neuron() = default;
};

class RecurrentRelu : public Neuron {

 public:
  float old_value;

  bool learning = true;

  void disable_learning();

  void enable_learning();

  void compute_gradient_of_all_synapses(std::vector<float> prediction_error_list);

  void update_value();

  float backward(float output_grad);

  float forward(float temp_value);

  Synapse* recurrent_synapse;

  RecurrentRelu(bool is_input, bool is_output);

  void fire();


};


class BiasNeuron : public Neuron {
 public:
  BiasNeuron();
  float backward(float output_grad);

  float forward(float temp_value);
};


class LinearNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  LinearNeuron(bool is_input, bool is_output);
};


class SigmoidNeuron : public Neuron {
 public:
  float backward(float output_grad);

  float forward(float temp_value);

  SigmoidNeuron(bool is_input, bool is_output);
};



#endif //INCLUDE_NN_SYNCED_NEURON_H_
