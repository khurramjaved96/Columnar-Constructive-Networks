//
// Created by Khurram Javed on 2022-11-26.
//

#ifndef INCLUDE_NN_LSTM_H_
#define INCLUDE_NN_LSTM_H_

#include "neuron.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "./dynamic_elem.h"
#include "./synapse.h"
#include "./message.h"
#include "./utils.h"

class LSTM : public Neuron{
 protected:
//  Variables for making predictions
  std::vector<float> w_i;
  std::vector<float> w_f;
  std::vector<float> w_g;
  std::vector<float> w_o;
  std::vector<float> input_means;
  std::vector<float> input_std;
  float epsilon;

  float decay_rate;

  bool update_statistics_flag;
  float std_cap;

  float u_i, u_f, u_g, u_o;
  float b_i, b_f, b_g, b_o;
  float c;
  float h;
  float old_c;
  float old_h;
  float i_val;
  float f;
  float g;
  float o;

//  Variables for computing the gradient
  std::vector<float> Hw_i;
  std::vector<float> Hw_f;
  std::vector<float> Hw_g;
  std::vector<float> Hw_o;

  std::vector<float> Cw_i;
  std::vector<float> Cw_f;
  std::vector<float> Cw_g;
  std::vector<float> Cw_o;

  float Hu_i, Hu_f, Hu_g, Hu_o;
  float Cu_i, Cu_f, Cu_g, Cu_o;

  float Hb_i, Hb_f, Hb_g, Hb_o;
  float Cb_i, Cb_f, Cb_g, Cb_o;


//  Variable for storing gradients
  std::vector<float> Gw_i;
  std::vector<float> Gw_f;
  std::vector<float> Gw_g;
  std::vector<float> Gw_o;

  float Gu_i, Gu_f, Gu_g, Gu_o;

  float Gb_i, Gb_f, Gb_g, Gb_o;

  // Variables for normalizing gradients
  std::vector<float> beta_2_w_i;
  std::vector<float> beta_2_w_f;
  std::vector<float> beta_2_w_g;
  std::vector<float> beta_2_w_o;

  float beta_2_u_i, beta_2_u_f, beta_2_u_g, beta_2_u_o;

  float beta_2_b_i, beta_2_b_f, beta_2_b_g, beta_2_b_o;

  int users;

  float copy_of_h;


 public:

  virtual void update_statistics();

  void set_update_statistics(bool val);

  std::vector<float> get_normalized_values();

  std::vector<Neuron *> incoming_neurons;

  virtual float get_value_without_sideeffects();

  int get_users();

  void decay_gradient(float decay_rate);

  void increment_user();

  void decrement_user();

  void update_value_delay();

  virtual void update_value_sync();

  void reset_state();

  void zero_grad();

  void accumulate_gradient(float incoming_grad);

  void print_gradients();

  float get_hidden_state();

  void update_weights(float step_size);

  void update_weights(float step_size, float error);

  void add_synapse(Neuron* s, float w_i, float w_f, float w_g, float w_o);

  float old_value;

  bool learning = true;

  virtual void compute_gradient_of_all_synapses();

  float backward(float output_grad) override;

  float forward(float temp_value) override;

  Synapse* recurrent_synapse;

  LSTM(float ui, float uf, float ug, float uo, float bi, float bf, float bg, float bo, float std_cap, float decay_rate);

  virtual void fire();


};

class LSTMNormalied : public LSTM{
  float state_mean;
  float state_variance;

  float get_normalized_hidden_state(float h );
 public:
  LSTMNormalied(float ui, float uf, float ug, float uo, float bi, float bf, float bg, float bo, float std_cap, float decay_rate);

  float get_value_without_sideeffects() override;

  void fire() override;

  void compute_gradient_of_all_synapses() override;

  void update_value_sync() override;

  void update_statistics() override;

};
#endif //INCLUDE_NN_LSTM_H_
