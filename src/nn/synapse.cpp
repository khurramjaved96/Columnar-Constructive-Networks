//
// Created by Khurram Javed on 2021-09-20.
//


#include "../../include/nn/synapse.h"
#include <math.h>
#include <vector>
#include <iostream>
#include "../../include/nn/neuron.h"
#include "../../include/nn/utils.h"

int64_t Synapse::synapse_id_generator = 0;

Synapse::Synapse(Neuron *input, Neuron *output, float w, float step_size) {
  references = 0;
  if(output->is_recurrent_neuron){
    this->is_recurrent_connection = true;
  }
  else {
    this->is_recurrent_connection = false;
  }
  if(input->id == output->id){
    RecurrentRelu* ref = dynamic_cast<RecurrentRelu*>(input);
    ref->recurrent_synapse = this;
  }

  input_neuron = input;
  input->increment_reference();
  output_neuron = output;
  output->increment_reference();
  this->gradient = 0;
  credit = 0;
  is_useless = false;
  age = 0;
  weight = w;
  this->step_size = step_size;
  if(input->id != output->id){
    this->increment_reference();
    input_neuron->outgoing_synapses.push_back(this);
  }
  this->increment_reference();
  output_neuron->incoming_synapses.push_back(this);
  this->idbd = false;
  this->id = synapse_id_generator;
  synapse_id_generator++;
  this->l2_norm_meta_gradient = 100;
  trace = 0;
  propagate_gradients = true;
  synapse_utility = 0;
  meta_step_size = 1e-4;
  if (input->is_input_neuron) {
    propagate_gradients = false;
  }

  this->TH = 0;
  utility_to_keep = 0.000001;
  disable_utility = false;
  this->trace_decay_rate = 0.999;
}
//

void Synapse::set_utility_to_keep(float util) {
  this->utility_to_keep = util;
}

float Synapse::get_utility_to_keep() {
  return this->utility_to_keep;
}

void Synapse::set_connected_to_recurrence(bool val) {
  this->is_recurrent_connection = val;
}


void Synapse::reset_trace() {
  this->trace = 0;
}

void Synapse::set_meta_step_size(float val) {
  this->meta_step_size = val;
}



void Synapse::update_utility() {

  float diff = this->output_neuron->value - this->output_neuron->forward(
      this->output_neuron->value_before_firing - this->input_neuron->value * this->weight);
//  0.999 is a hyper-parameter.
  if (!this->disable_utility) {
    this->synapse_local_utility_trace = this->trace_decay_rate * this->synapse_local_utility_trace + (1-this->trace_decay_rate) * std::abs(diff);
    this->synapse_utility =
        (synapse_local_utility_trace * this->output_neuron->neuron_utility)
            / (this->output_neuron->sum_of_utility_traces + 1e-10);
    if (this->synapse_utility > this->utility_to_keep) {
      this->synapse_utility_to_distribute = this->synapse_utility - this->utility_to_keep;
      this->synapse_utility = this->utility_to_keep;
    } else {
      this->synapse_utility_to_distribute = 0;
    }
  } else {
    this->synapse_utility = 0;
    this->synapse_utility_to_distribute = 0;
    this->synapse_local_utility_trace = 0;
  }
}

/**
 * Calculate and set credit based on gradients in the current synapse.
 */
void Synapse::assign_credit() {

  this->credit = this->gradient;
//  this->trace = this->trace * gamma * lambda +
//        this->gradient;
//
//
////  this->tidbd_old_activation = this->weight_assignment_past_activations.gradient_activation;
////  this->tidbd_old_error = this->grad_queue_weight_assignment.error;
//
//  this->credit = this->trace * prediction_error;
////  std::cout << "Credit = " << this->credit << std::endl;
}

void Synapse::block_gradients() {
  propagate_gradients = false;
}

bool Synapse::get_recurrent_status() {
  return is_recurrent_connection;
}

void Synapse::turn_on_idbd() {
  this->idbd = true;
  this->log_step_size_tidbd = log(this->step_size);
  this->h_tidbd = 0;
  this->step_size = exp(this->log_step_size_tidbd);
}

void Synapse::turn_off_idbd() {
  this->idbd = false;
}

void Synapse::update_weight() {
//
  if (this->idbd) {
    float meta_grad = this->tidbd_old_error * this->trace * this->h_tidbd;
    this->l2_norm_meta_gradient = this->l2_norm_meta_gradient * 0.99 + (1 - 0.99) * (meta_grad * meta_grad);
    if (age > 1000) {
      this->log_step_size_tidbd += this->meta_step_size * meta_grad / (sqrt(this->l2_norm_meta_gradient) + 1e-8);
      this->log_step_size_tidbd = max(this->log_step_size_tidbd, -15);
      this->log_step_size_tidbd = min(this->log_step_size_tidbd, -3);
      this->step_size = exp(this->log_step_size_tidbd);
      this->weight -= (this->step_size * this->credit);
      if ((1 - this->step_size * this->tidbd_old_activation * this->trace) > 0) {
        this->h_tidbd =
            this->h_tidbd * (1 - this->step_size * this->tidbd_old_activation * this->trace) +
                this->step_size * this->trace * this->tidbd_old_error;
//        std::cout << "Decay rate " << (1 - this->step_size * this->tidbd_old_activation * this->trace) << std::endl;
      } else {
        this->h_tidbd = this->step_size * this->trace * this->tidbd_old_error;
      }
    }

  } else {
//    std::cout << "Updating weight\n";
    this->weight -= (this->step_size * this->credit);
    if(this->input_neuron->id == this->output_neuron->id){
      if(this->weight >= 0.999){
        this->weight = 0.99;
      }
      if(this->weight < 0)
        this->weight = 0;
    }
  }
  this->credit = 0;
  this->gradient = 0;
}

