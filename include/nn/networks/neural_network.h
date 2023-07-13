//
// Created by Khurram Javed on 2021-09-20.
//

#ifndef INCLUDE_NN_NETWORKS_NETWORKSYNCED_H_
#define INCLUDE_NN_NETWORKS_NETWORKSYNCED_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synapse.h"
#include "../neuron.h"
#include "../dynamic_elem.h"

class NeuralNetwork {
 protected:
  int64_t time_step;
  std::mt19937 mt;

 public:
  std::vector<Neuron *> all_neurons;
  std::vector<Neuron *> output_neurons;
  std::vector<Neuron *> input_neurons;
  std::vector<RecurrentRelu *> recurrent_features;
  std::vector<Synapse *> all_synapses;
  std::vector<Synapse *> output_synapses;
  std::vector<dynamic_elem *> all_heap_elements;


  void collect_garbage();

  NeuralNetwork();

  ~NeuralNetwork();

  int64_t get_timestep();

  void set_input_values(std::vector<float> const &input_values);

  void step();

  std::vector<float> read_output_values();

  std::vector<float> read_all_values();

  std::vector<float> read_all_weights();

  float introduce_targets(std::vector<float> targets);

//  float introduce_targets(std::vector<float> targets, float gamma, float lambda);

//  float introduce_targets(float targets, float gamma, float lambda, std::vector<bool> no_grad);

  std::vector<float> forward_pass_without_side_effects(std::vector<float> input_vector);

  int get_input_size();

  void print_synapse_status();

  void print_neuron_status();

  int get_total_synapses();

  int get_total_neurons();

  void reset_trace();

  void viz_graph();

  std::string get_viz_graph();

//    virtual void add_feature() = 0;
};

#endif //INCLUDE_NN_NETWORKS_NETWORKSYNCED_H_
