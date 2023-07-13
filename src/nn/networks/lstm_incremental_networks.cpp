//
// Created by Khurram Javed on 2022-01-17.
//

#include "../../../include/nn/networks/lstm_incremental_networks.h"
#include "../../../include/nn/neuron.h"
#include <algorithm>
#include <cmath>
#include <iostream>
//

IncrementalNetworks::IncrementalNetworks(float step_size, int seed,
                                         int no_of_input_features,
                                         int total_targets,
                                         int total_recurrent_features,
                                         int layer_size, float std_cap,
                                         float decay_rate) {
  this->decay_rate = decay_rate;
  this->layer_size = layer_size;
  this->step_size = step_size;
  this->std_cap = std_cap;
  this->mt.seed(seed);
  std::mt19937 second_mt(seed);
  std::uniform_real_distribution<float> weight_sampler(-0.01, 0.01);
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  std::uniform_int_distribution<int> index_sampler(
      0, no_of_input_features + total_recurrent_features - 1);

  for (int i = 0; i < no_of_input_features; i++) {
    LinearNeuron n(true, false);
    this->input_neurons.push_back(n);
  }

  for (int i = 0; i < total_recurrent_features; i++) {
    //    std::cout << "Recurrent feature no "<< i << std::endl;
    LSTM *lstm_neuron = new LSTM(
        weight_sampler(mt), weight_sampler(mt), weight_sampler(mt),
        weight_sampler(mt), weight_sampler(mt), weight_sampler(mt),
        weight_sampler(mt), weight_sampler(mt), this->std_cap, decay_rate);
    indexes_lstm_cells.push_back(i);
    this->LSTM_neurons.push_back(lstm_neuron);
  }

  for (int counter = 0; counter < total_recurrent_features; counter++) {
    int layer_no = counter / layer_size;
    // int max_connections = (layer_no * layer_size) + no_of_input_features;
    // //dense
    int max_connections = no_of_input_features;
    int incoming_features = 0;
    std::vector<int> map_index(no_of_input_features + total_recurrent_features,
                               0);
    int counter_temp_temp = 0;
    int temp_counter = 0;
    int total_features = 0;
    while (temp_counter < 50000) {
//    while (total_features < 20) {
      temp_counter++;
      //    while (counter_temp_temp < 4000) {
      counter_temp_temp++;
      int index = index_sampler(second_mt);
      if (map_index[index] == 0) {
        map_index[index] = 1;

        if (index < no_of_input_features) {
          total_features++;
          //          std::cout << "Inp " << index << "\t" << counter <<
          //          std::endl;
          incoming_features++;
          Neuron *neuron_ref = &this->input_neurons[index];
          LSTM_neurons[counter]->add_synapse(
              neuron_ref, weight_sampler(mt), weight_sampler(mt),
              weight_sampler(mt), weight_sampler(mt));
        } else {
          index = index - no_of_input_features;
          int new_layer_no = index / layer_size;
          if (new_layer_no < layer_no) {
            total_features++;

            //            std::cout << index << "\t" << counter << std::endl;
            incoming_features++;
            Neuron *neuron_ref = this->LSTM_neurons[index];
            // TODO making it single layer
            // Neuron *neuron_ref = &this->input_neurons[index];
            LSTM_neurons[counter]->add_synapse(
                neuron_ref, weight_sampler(mt), weight_sampler(mt),
                weight_sampler(mt), weight_sampler(mt));
          }
        }
      }
    }
    std::cout << "Total features = " << total_features << std::endl;
  }
  for (int counter = 0; counter < this->LSTM_neurons.size(); counter++) {
    for (int inner_counter = 0;
         inner_counter < this->LSTM_neurons[counter]->incoming_neurons.size();
         inner_counter++) {
//            std::cout
//                <<
//                this->LSTM_neurons[counter]->incoming_neurons[inner_counter]->id
//                << "\tto\t" << this->LSTM_neurons[counter]->id << std::endl;
    }
  }
  //  exit(1);

  predictions = 0;
  bias = 0;
  bias_gradients = 0;
  bias_beta_2 = 1;
  for (int j = 0; j < this->LSTM_neurons.size(); j++) {
    prediction_weights.push_back(0);
    prediction_weights_gradient.push_back(0);
    avg_feature_value.push_back(0);
    feature_mean.push_back(0);
    feature_std.push_back(1);
    prediction_weights_beta_2.push_back(1);
  }
}

IncrementalNetworks::IncrementalNetworks() {}

Snap1::Snap1(float step_size, int seed, int no_of_input_features,
             int total_targets, int total_recurrent_features, int layer_size,
             float std_cap) {
  this->layer_size = layer_size;
  this->step_size = step_size;
  this->std_cap = std_cap;
  this->mt.seed(seed);
  std::mt19937 second_mt(seed);
  std::uniform_real_distribution<float> weight_sampler(-0.1, 0.1);
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  std::uniform_int_distribution<int> index_sampler(
      0, no_of_input_features + total_recurrent_features - 1);

  for (int i = 0; i < no_of_input_features; i++) {
    LinearNeuron n(true, false);
    this->input_neurons.push_back(n);
  }

  for (int i = 0; i < total_recurrent_features; i++) {
    //    std::cout << "Recurrent feature no "<< i << std::endl;
    LSTM *lstm_neuron = new LSTM(
        weight_sampler(mt), weight_sampler(mt), weight_sampler(mt),
        weight_sampler(mt), weight_sampler(mt), weight_sampler(mt),
        weight_sampler(mt), weight_sampler(mt), this->std_cap, 0.99999);

    indexes_lstm_cells.push_back(i);
    this->LSTM_neurons.push_back(lstm_neuron);
  }

  for (int counter = 0; counter < total_recurrent_features; counter++) {
    int incoming_features = 0;
    std::vector<int> map_index(no_of_input_features + total_recurrent_features,
                               0);
    int counter_temp_temp = 0;
    int temp_counter = 0;
    while (temp_counter < 10000) {
      temp_counter++;
      //    while (counter_temp_temp < 4000) {
      counter_temp_temp++;
      int index = index_sampler(second_mt);
      if (map_index[index] == 0) {
        map_index[index] = 1;
        if (index < no_of_input_features) {
                    std::cout << "Inp " << index << "\t" << counter <<
                    std::endl;
          incoming_features++;
          Neuron *neuron_ref = &this->input_neurons[index];
          LSTM_neurons[counter]->add_synapse(
              neuron_ref, weight_sampler(mt), weight_sampler(mt),
              weight_sampler(mt), weight_sampler(mt));
        } else {
          index = index - no_of_input_features;
          incoming_features++;
          Neuron *neuron_ref = this->LSTM_neurons[index];
          // TODO making it single layer
          // Neuron *neuron_ref = &this->input_neurons[index];
          LSTM_neurons[counter]->add_synapse(
              neuron_ref, weight_sampler(mt), weight_sampler(mt),
              weight_sampler(mt), weight_sampler(mt));
          //          }
        }
      }
    }
    int sum_temp = 0;
    for (int i = 0; i < map_index.size(); i++)
      sum_temp += map_index[i];
    std::cout << "Sum temp = " << sum_temp << std::endl;
  }

  predictions = 0;
  bias = 0;
  bias_gradients = 0;
  for (int j = 0; j < this->LSTM_neurons.size(); j++) {
    prediction_weights.push_back(0);
    prediction_weights_gradient.push_back(0);
    avg_feature_value.push_back(0);
    feature_mean.push_back(0);
    feature_std.push_back(1);
  }
}

void IncrementalNetworks::print_features_stats() {
  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    std::cout << "Counter = " << counter << std::endl;
    std::cout << "Feature mean = " << feature_mean[counter] << std::endl;
    std::cout << "Feature std = " << feature_std[counter] << std::endl;
    std::cout << "feature value = " << LSTM_neurons[counter]->value
              << std::endl;
  }
}

float IncrementalNetworks::forward(std::vector<float> inputs) {

  for (int i = 0; i < inputs.size(); i++) {
    this->input_neurons.at(i).value = inputs[i];
  }

  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter]->update_value_sync();
  }

  //  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
  //    LSTM_neurons[counter]->compute_gradient_of_all_synapses();
  //  }

  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter]->fire();
    avg_feature_value[counter] =
        avg_feature_value[counter] * this->decay_rate +
        (1 - this->decay_rate) *
            (LSTM_neurons[counter]->value - feature_mean[counter]);
  }

  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    feature_mean[counter] = feature_mean[counter] * this->decay_rate + (1 - this->decay_rate) * LSTM_neurons[counter]->value;
    //    std::cout << "Feature mean = " << feature_mean[counter] << std::endl;
    if (std::isnan(feature_mean[counter])) {
      std::cout << "Feature mean = " << feature_mean[counter] << std::endl;
      std::cout << "Feature std = " << feature_std[counter] << std::endl;
      std::cout << "feature value = " << LSTM_neurons[counter]->value
                << std::endl;
      exit(1);
    }
    float temp = (feature_mean[counter] - LSTM_neurons[counter]->value);
    feature_std[counter] = feature_std[counter] * this->decay_rate +
                           (1 - decay_rate) * temp * temp;
    if (feature_std[counter] < this->std_cap)
      feature_std[counter] = this->std_cap;
  }

  predictions = 0;
  for (int i = 0; i < prediction_weights.size(); i++) {
    predictions += prediction_weights[i] *
                   (this->LSTM_neurons[i]->value - feature_mean[i]) /
                   sqrt(feature_std[i]);
  }
  predictions += bias;
  return predictions;
}

float IncrementalNetworks::get_target_without_sideeffects(
    std::vector<float> inputs) {
  //  Backup old values
  std::vector<float> backup_vales;
  for (int i = 0; i < inputs.size(); i++) {
    backup_vales.push_back(this->input_neurons[i].value);
    this->input_neurons[i].value = inputs[i];
  }

  //  Get hidden state without side-effects
  std::vector<float> hidden_state;
  for (int i = 0; i < LSTM_neurons.size(); i++) {
    hidden_state.push_back(LSTM_neurons[i]->get_value_without_sideeffects());
  }

  //  Compute prediction
  float temp_prediction = 0;
  for (int i = 0; i < prediction_weights.size(); i++) {
    temp_prediction +=
        prediction_weights[i] *
        ((hidden_state[i] - feature_mean[i]) / sqrt(feature_std[i]));
  }
  temp_prediction += bias;

  //  Restore values
  for (int i = 0; i < inputs.size(); i++) {
    this->input_neurons[i].value = backup_vales[i];
  }
  return temp_prediction;
}

void IncrementalNetworks::reset_state() {
  for (int i = 0; i < LSTM_neurons.size(); i++) {
    LSTM_neurons[i]->reset_state();
  }
}

void IncrementalNetworks::zero_grad() {
  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter]->zero_grad();
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights_gradient[index] = 0;
  }

  bias_gradients = 0;
}

void IncrementalNetworks::decay_gradient(float decay_rate) {
  for (int counter = 0; counter < LSTM_neurons.size(); counter++) {
    LSTM_neurons[counter]->decay_gradient(decay_rate);
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights_gradient[index] *= decay_rate;
  }

  bias_gradients *= decay_rate;
}

std::vector<float> IncrementalNetworks::get_prediction_weights() {
  std::vector<float> my_vec;
  my_vec.reserve(prediction_weights.size());
  for (int index = 0; index < prediction_weights.size(); index++) {
    my_vec.push_back(prediction_weights[index]);
  }
  return my_vec;
}

std::vector<float> IncrementalNetworks::get_prediction_gradients() {
  std::vector<float> my_vec;
  my_vec.reserve(prediction_weights_gradient.size());
  for (int index = 0; index < prediction_weights_gradient.size(); index++) {
    my_vec.push_back(prediction_weights_gradient[index]);
  }
  return my_vec;
}

std::vector<float> IncrementalNetworks::get_state() {
  std::vector<float> my_vec;
  my_vec.reserve(LSTM_neurons.size());
  for (int index = 0; index < LSTM_neurons.size(); index++) {
    my_vec.push_back(LSTM_neurons[index]->value);
  }
  return my_vec;
}

std::vector<float> IncrementalNetworks::get_normalized_state() {
  std::vector<float> my_vec;
  my_vec.reserve(LSTM_neurons.size());
  for (int i = 0; i < LSTM_neurons.size(); i++) {
    my_vec.push_back((this->LSTM_neurons[i]->value - feature_mean[i]) /
                     sqrt(feature_std[i]));
  }
  return my_vec;
}

void IncrementalNetworks::backward(int layer) {

  //  Update the prediction weights
  for (int index = 0; index < LSTM_neurons.size(); index++) {
    if ((layer)*layer_size <= index && index < (layer + 1) * layer_size) {
      LSTM_neurons[index]->compute_gradient_of_all_synapses();
      float gradient = prediction_weights[index] / sqrt(feature_std[index]);
      LSTM_neurons[index]->accumulate_gradient(gradient);
    }
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights_gradient[index] +=
        (LSTM_neurons[index]->value - feature_mean[index]) /
        sqrt(feature_std[index]);
  }

  bias_gradients += 1;
}

void Snap1::backward(int layer) {

  //  Update the prediction weights
  for (int index = 0; index < LSTM_neurons.size(); index++) {
    LSTM_neurons[index]->compute_gradient_of_all_synapses();
    float gradient = prediction_weights[index] / sqrt(feature_std[index]);
    LSTM_neurons[index]->accumulate_gradient(gradient);
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights_gradient[index] +=
        (LSTM_neurons[index]->value - feature_mean[index]) /
        sqrt(feature_std[index]);
  }

  bias_gradients += 1;
}

void IncrementalNetworks::update_parameters(int layer, float error) {

  float b2_decay = 0.9999;
  float epsilon = 1e-8;
  for (int index = 0; index < LSTM_neurons.size(); index++) {
    if ((layer)*layer_size <= index && index < (layer + 1) * layer_size) {
      LSTM_neurons[index]->update_weights(step_size, error);
    } else if (index < (layer)*layer_size) {
      LSTM_neurons[index]->set_update_statistics(false);
    }
  }

  float total_features_for_prediction = (layer + 1) * layer_size;
  float temp_step_size = step_size/total_features_for_prediction;
  for (int index = 0; index < LSTM_neurons.size(); index++) {
    if (index < (layer + 1) * layer_size) {

      //      We normalize the step-size by total out-going weights that are
      //      being updated; this is necessary because as we increase the number
      //      of features in the last layer, the optimal step-size would change.
      // Since all features are normalized to have a mean of zero and variance
      // of one, it is sufficient to just divide by total number of features
      prediction_weights_beta_2[index] = prediction_weights_beta_2[index]*b2_decay + (1-b2_decay)*(prediction_weights_gradient[index] * error)*(prediction_weights_gradient[index] * error);
      prediction_weights[index] += prediction_weights_gradient[index] * error *
                                   (temp_step_size / (sqrt(prediction_weights_beta_2[index]) + epsilon));
    }
  }
  bias_beta_2 = bias_beta_2*b2_decay + (1-b2_decay)*(error*bias_gradients)*(error*bias_gradients);
  bias +=  (temp_step_size / (sqrt(bias_beta_2) + epsilon)) * bias_gradients * error;
}

void Snap1::update_parameters(int layer, float error) {

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    LSTM_neurons[index]->update_weights(step_size, error);
  }

  float total_features_for_prediction = (layer + 1) * layer_size;
  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights[index] += prediction_weights_gradient[index] * error *
                                 (step_size / total_features_for_prediction);
  }
  bias += error * step_size * bias_gradients;
}

//
std::vector<float> IncrementalNetworks::real_all_running_mean() {
  std::vector<float> output_val;
  output_val.reserve(this->input_neurons.size() + this->LSTM_neurons.size());
  //  Store input values
  for (auto n : this->input_neurons)
    output_val.push_back(n.running_mean);
  //  Store other values
  for (auto n : this->LSTM_neurons)
    output_val.push_back(n->running_mean);
  return output_val;
}

std::vector<float> IncrementalNetworks::read_all_running_variance() {
  std::vector<float> output_val;
  output_val.reserve(this->input_neurons.size() + this->LSTM_neurons.size());
  //  Store input values
  for (auto n : this->input_neurons)
    output_val.push_back(n.running_variance);
  //  Store other values
  for (auto n : this->LSTM_neurons)
    output_val.push_back(n->running_variance);
  return output_val;
}

void IncrementalNetworks::update_parameters_no_freeze(float error) {

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    LSTM_neurons[index]->update_weights(step_size, error);
  }

  for (int index = 0; index < LSTM_neurons.size(); index++) {
    prediction_weights[index] +=
        prediction_weights_gradient[index] * error * step_size;
  }

  bias += error * step_size * bias_gradients;
}

float IncrementalNetworks::read_output_values() { return predictions; }

IncrementalNetworks::~IncrementalNetworks(){};

NormalizedIncrementalNetworks::NormalizedIncrementalNetworks(
    float step_size, int seed, int no_of_input_features, int total_targets,
    int total_recurrent_features, int layer_size, float std_cap,
    float decay_rate) {
  this->layer_size = layer_size;
  this->decay_rate = decay_rate;
  this->step_size = step_size;
  this->std_cap = std_cap;
  this->mt.seed(seed);
  std::mt19937 second_mt(seed);
  std::uniform_real_distribution<float> weight_sampler(-0.1, 0.1);
  std::uniform_real_distribution<float> prob_sampler(0, 1);
  std::uniform_int_distribution<int> index_sampler(
      0, no_of_input_features + total_recurrent_features - 1);

  for (int i = 0; i < no_of_input_features; i++) {
    LinearNeuron n(true, false);
    this->input_neurons.push_back(n);
  }

  for (int i = 0; i < total_recurrent_features; i++) {
    //    std::cout << "Recurrent feature no "<< i << std::endl;
    LSTM *lstm_neuron = new LSTMNormalied(
        weight_sampler(mt), weight_sampler(mt), weight_sampler(mt),
        weight_sampler(mt), weight_sampler(mt), weight_sampler(mt),
        weight_sampler(mt), weight_sampler(mt), this->std_cap,
        this->decay_rate);
    //    for (int counter = 0; counter < this->input_neurons.size(); counter++)
    //    {
    //      Neuron *neuron_ref = &this->input_neurons[counter];
    //      lstm_neuron.add_synapse(neuron_ref,
    //                              weight_sampler(mt),
    //                              weight_sampler(mt),
    //                              weight_sampler(mt),
    //                              weight_sampler(mt));
    //    }
    indexes_lstm_cells.push_back(i);
    this->LSTM_neurons.push_back(lstm_neuron);
  }

  for (int counter = 0; counter < total_recurrent_features; counter++) {
    int layer_no = counter / layer_size;
    // int max_connections = (layer_no * layer_size) + no_of_input_features;
    // //dense
    int max_connections = no_of_input_features;
    int incoming_features = 0;
    std::vector<int> map_index(no_of_input_features + total_recurrent_features,
                               0);
    int counter_temp_temp = 0;
    int temp_counter = 0;
    while (temp_counter < 10000) {
      temp_counter++;
      //    while (counter_temp_temp < 4000) {
      counter_temp_temp++;
      int index = index_sampler(second_mt);
      if (map_index[index] == 0) {
        map_index[index] = 1;
        if (index < no_of_input_features) {
          //          std::cout << "Inp " << index << "\t" << counter <<
          //          std::endl;
          incoming_features++;
          Neuron *neuron_ref = &this->input_neurons[index];
          LSTM_neurons[counter]->add_synapse(
              neuron_ref, weight_sampler(mt), weight_sampler(mt),
              weight_sampler(mt), weight_sampler(mt));
        } else {
          index = index - no_of_input_features;
          int new_layer_no = index / layer_size;
          if (new_layer_no < layer_no) {
            //            std::cout << index << "\t" << counter << std::endl;
            incoming_features++;
            Neuron *neuron_ref = this->LSTM_neurons[index];
            // TODO making it single layer
            // Neuron *neuron_ref = &this->input_neurons[index];
            LSTM_neurons[counter]->add_synapse(
                neuron_ref, weight_sampler(mt), weight_sampler(mt),
                weight_sampler(mt), weight_sampler(mt));
          }
        }
      }
    }
  }
  for (int counter = 0; counter < this->LSTM_neurons.size(); counter++) {
    for (int inner_counter = 0;
         inner_counter < this->LSTM_neurons[counter]->incoming_neurons.size();
         inner_counter++) {
      std::cout
          << this->LSTM_neurons[counter]->incoming_neurons[inner_counter]->id
          << "\tto\t" << this->LSTM_neurons[counter]->id << std::endl;
    }
  }
  //  exit(1);

  predictions = 0;
  bias = 0;
  bias_gradients = 0;
  for (int j = 0; j < this->LSTM_neurons.size(); j++) {
    prediction_weights.push_back(0);
    prediction_weights_gradient.push_back(0);
    avg_feature_value.push_back(0);
    feature_mean.push_back(0);
    feature_std.push_back(1);
  }
}