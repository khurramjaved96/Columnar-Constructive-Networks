//
// Created by Khurram Javed on 2021-03-16.
//

#include "../../include/nn/neuron.h"
#include <assert.h>
#include <iostream>
#include <utility>
#include <random>
#include <algorithm>
#include <vector>
#include "../../include/utils.h"
#include "../../include/nn/lstm.h"

Neuron::Neuron(bool is_input, bool is_output) {
  value = 0;
  frozen = false;
  value_before_firing = 0;
  id = neuron_id_generator;
  useless_neuron = false;
  neuron_id_generator++;
  this->is_output_neuron = is_output;
  is_input_neuron = is_input;
  neuron_age = 0;
  references = 0;
  neuron_utility = 0;
  drinking_age = 5000;
  is_bias_unit = false;
  is_recurrent_neuron = false;
  neuron_utility_trace_decay_rate = 0.9999;
  drinking_age = 5.0 / (1.0 - neuron_utility_trace_decay_rate);
}

void Neuron::set_layer_number(int layer) {
  this->layer_number = layer;
}

int Neuron::get_layer_number() {
  return this->layer_number;
}

void Neuron::update_utility() {

  for(auto synapse: this->outgoing_synapses) {
    this->neuron_utility = this->neuron_utility * this->neuron_utility_trace_decay_rate
        + std::abs(this->value * synapse->weight) * (1 - neuron_utility_trace_decay_rate);
  }

}

float Neuron::get_utility() {
  return this->neuron_utility;
}

void Neuron::fire() {
  this->neuron_age++;
  this->value = this->forward(value_before_firing);
}

void RecurrentRelu::fire() {
  this->neuron_age++;
  this->old_value = this->value;
  this->value = this->forward(value_before_firing);
}

bool Neuron::is_mature() {
//  std::cout << "Neuron age \t Drinking age\n";
//  std::cout << this->neuron_age << "\t" << this->drinking_age << std::endl;
  if (this->neuron_age > this->drinking_age)
    return true;
  return false;
}

void RecurrentRelu::compute_gradient_of_all_synapses(std::vector<float> prediction_error_list) {
//  First we get error from the output node
  float incoming_gradient_sum = 0;
  int counter = 0;
  for (auto synapse: this->outgoing_synapses) {
    if(synapse->output_neuron->is_output_neuron) {
      synapse->gradient = this->value * prediction_error_list[counter];
      incoming_gradient_sum +=
          prediction_error_list[counter] * synapse->output_neuron->backward(synapse->output_neuron->value)
              * synapse->weight;
      counter++;
    }
  }

//  Then we update the trace value for RTRL computation of all the parameters
  for (auto synapse: this->incoming_synapses) {
    if (synapse->input_neuron->id == synapse->output_neuron->id) {
      synapse->TH = this->backward(this->value) * (this->old_value + this->recurrent_synapse->weight * synapse->TH);
    } else {
      synapse->TH =
          this->backward(this->value) * (synapse->input_neuron->value + this->recurrent_synapse->weight * synapse->TH);
    }
    synapse->gradient = synapse->TH * incoming_gradient_sum;
  }
//    Finally, we use the updated TH value to compute the gradient

}

void Neuron::update_value() {
//  std::cout << "Updating value of non-recurrent neuron : " << this->id << "\n";
  this->value_before_firing = 0;

//  Age our neuron like a fine wine and set the next values of our neuron.
  for (auto &it : this->incoming_synapses) {
    it->age++;
    this->value_before_firing += it->weight * it->input_neuron->value;
  }
}

void RecurrentRelu::disable_learning() {
  this->learning = false;
  for (auto synapse: this->incoming_synapses) {
    synapse->TH = 0;
  }
}

void RecurrentRelu::enable_learning() {
  this->learning = true;
}

void RecurrentRelu::update_value() {
//  std::cout << "Updating value of recurrent neuron: " << this->id << "\n";
  this->value_before_firing = 0;

//  Age our neuron like a fine wine and set the next values of our neuron.
  for (auto &it : this->incoming_synapses) {
    it->age++;
    this->value_before_firing += it->weight * it->input_neuron->value;
  }
}

/**
 * Introduce a target to a neuron and calculate its error.
 * In this case, target should be our TD target, and the neuron should be an outgoing neuron.
 * @param target: target gradient_activation to calculate our error.
 * @param time_step: time step that we calculate this error. Use for backprop purposes.
 * @return: squared error
 */
float Neuron::introduce_targets(float target) {

  float error = this->value - target;
  return error * error;
}

float LinearNeuron::forward(float temp_value) {
  return temp_value;
}

float LinearNeuron::backward(float post_activation) {
  return 1;
}

float BiasNeuron::forward(float temp_value) {
  return 1;
}

float BiasNeuron::backward(float output_grad) {
  return 0;
}

float RecurrentRelu::forward(float temp_value) {
  if (temp_value <= 0)
    return 0;
  return temp_value;
}

float RecurrentRelu::backward(float post_activation) {
  if (post_activation > 0) {
    return 1;
  }
  return 0;
}

RecurrentRelu::RecurrentRelu(bool is_input, bool is_output) : Neuron(is_input, is_output) {
  this->old_value = 0;
  this->is_recurrent_neuron = true;
}

BiasNeuron::BiasNeuron() : Neuron(false, false) {
  this->is_bias_unit = true;
}

LinearNeuron::LinearNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {}

SigmoidNeuron::SigmoidNeuron(bool is_input, bool is_output) : Neuron(is_input, is_output) {}

float SigmoidNeuron::forward(float temp_value) {

  return sigmoid(temp_value);
}

float SigmoidNeuron::backward(float post_activation) {
  return post_activation * (1 - post_activation);
}


std::mt19937 Neuron::gen = std::mt19937(0);

int64_t Neuron::neuron_id_generator = 0;


