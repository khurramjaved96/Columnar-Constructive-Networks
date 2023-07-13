//
// Created by Khurram Javed on 2021-09-20.
//

#ifndef INCLUDE_NN_SYNCED_SYNAPSE_H_
#define INCLUDE_NN_SYNCED_SYNAPSE_H_


#include <vector>
#include <queue>
#include <utility>
#include "./dynamic_elem.h"
#include "./message.h"

class Neuron;

class Synapse : public dynamic_elem {
  bool is_recurrent_connection;

 public:
  void set_meta_step_size(float);
  float meta_step_size;
  static int64_t synapse_id_generator;
  int64_t id;

  bool is_useless;
  int age;
  float weight;
  float credit;
  float trace;
  float step_size;

//  Parameter only used for recurrent neurons
  float TH;
  float gradient;
  float utility_to_keep;
  float synapse_utility;
  float synapse_utility_to_distribute;
  float synapse_local_utility_trace = 0;
  float tidbd_old_activation;
  float tidbd_old_error;
  bool propagate_gradients;
  float l2_norm_meta_gradient;
  float log_step_size_tidbd;
  float h_tidbd;
  bool idbd;
  float trace_decay_rate = 0.9999;
  bool disable_utility; //Also disables pruning

  void set_connected_to_recurrence(bool);

  bool get_recurrent_status();

  Neuron *input_neuron;
  Neuron *output_neuron;

  explicit Synapse(Neuron *input, Neuron *output, float w, float step_size);


  void block_gradients();

  void set_shadow_weight(bool);

  void update_weight();

  void turn_on_idbd();

  void turn_off_idbd();

  void update_utility();

  void memory_leak_patch();

  void assign_credit();

  void reset_trace();

  void set_utility_to_keep(float);

  float get_utility_to_keep();

  ~Synapse() = default;
};






#endif //INCLUDE_NN_SYNCED_SYNAPSE_H_
