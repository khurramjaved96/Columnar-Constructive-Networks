//
// Created by Khurram Javed on 2022-01-16.
//

#include "../../include/nn/lstm.h"
#include "../../include/nn/neuron.h"
#include "../../include/nn/utils.h"
#include "../../include/utils.h"
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

LSTM::LSTM(float ui, float uf, float ug, float uo, float bi, float bf, float bg,
           float bo, float std_cap, float decay_rate)
    : Neuron(false, false) {
  this->std_cap = std_cap;
  this->update_statistics_flag = true;
  this->epsilon = 1e-8;
  Hu_i = Hu_f = Hu_g = Hu_o = 0;
  Cu_i = Cu_f = Cu_g = Cu_o = 0;
  Hb_i = Hb_f = Hb_g = Hb_o = 0;
  Cb_i = Cb_f = Cb_g = Cb_o = 0;
  //  Gradients
  Gb_i = Gb_f = Gb_g = Gb_o = 0;
  Gu_i = Gu_f = Gu_g = Gu_o = 0;

  // RMSProp normalization terms
  beta_2_b_i = beta_2_b_f = beta_2_b_g = beta_2_b_o = 1;
  beta_2_u_i = beta_2_u_f = beta_2_u_g = beta_2_u_o = 1;

  //  Order: i|f|g|o
  u_i = ui;
  u_f = uf;
  u_g = ug;
  u_o = uo;

  b_i = bi;
  b_f = bf;
  b_g = bg;
  b_o = bo;
  this->decay_rate = decay_rate;
  this->value = 0;
}

float LSTM::forward(float temp_value) { return temp_value; }

float LSTM::backward(float output_grad) { return 1; }

void LSTM::set_update_statistics(bool val) {
  this->update_statistics_flag = val;
}

void LSTM::print_gradients() {
  std::cout << "W_i gradients\n";
  for (int i = 0; i < this->w_i.size(); i++) {
    std::cout << Gw_i[i] << ", ";
  }
  std::cout << "\n";

  std::cout << "W_f gradients\n";
  for (int i = 0; i < this->w_f.size(); i++) {
    std::cout << Gw_f[i] << ", ";
  }
  std::cout << "\n";

  std::cout << "W_g gradients\n";
  for (int i = 0; i < this->w_g.size(); i++) {
    std::cout << Gw_g[i] << ", ";
  }
  std::cout << "\n";

  std::cout << "W_o gradients\n";
  for (int i = 0; i < this->w_g.size(); i++) {
    std::cout << Gw_o[i] << ", ";
  }
  std::cout << "\n";

  std::cout << "Bias \n";
  //  Order: i|f|g|o
  std::cout << Gb_i << " " << Gb_f << " " << Gb_g << " " << Gb_o << std::endl;

  std::cout << "Recurrent connection \n";
  //  Order: i|f|g|o
  std::cout << Gu_i << " " << Gu_f << " " << Gu_g << " " << Gu_o << std::endl;
}

void LSTM::fire() {

  this->value = h;
  this->neuron_age++;
}

void LSTM::compute_gradient_of_all_synapses() {
  std::vector<float> x = get_normalized_values();
  int n = this->incoming_neurons.size();

  //  Computing terms that are reused to save flops

  float LHS_g = (1 - (g * g));
  float LHS_f = f * (1 - f);
  float LHS_i = i_val * (1 - i_val);
  float LHS_o = o * (1 - o);
  float tanh_of_c = tanh(c);

  //  Section 1.1

  for (int i = 0; i < n; i++) {
    float dg_dWi = LHS_g * (u_g * Hw_i[i]);
    float df_dWi = LHS_f * (u_f * Hw_i[i]);
    float di_dWi = LHS_i * (x[i] + u_i * Hw_i[i]);
    float do_dWi = LHS_o * (u_o * Hw_i[i]);
    Cw_i[i] = f * Cw_i[i] + old_c * df_dWi + i_val * dg_dWi + g * di_dWi;
    Hw_i[i] = o * (1 - tanh_of_c * tanh_of_c) * Cw_i[i] + tanh_of_c * do_dWi;
  }

  //  Section 1.4
  for (int i = 0; i < n; i++) {
    float dg_dWF = LHS_g * (u_g * Hw_f[i]);
    float df_dWF = LHS_f * (x[i] + u_f * Hw_f[i]);
    float di_dWF = LHS_i * (u_i * Hw_f[i]);
    float do_dWF = LHS_o * (u_o * Hw_f[i]);
    Cw_f[i] = f * Cw_f[i] + old_c * df_dWF + i_val * dg_dWF + g * di_dWF;
    Hw_f[i] = o * (1 - tanh_of_c * tanh_of_c) * Cw_f[i] + tanh_of_c * do_dWF;
  }

  //  Section 1.5
  for (int i = 0; i < n; i++) {
    float dg_dWo = LHS_g * (u_g * Hw_o[i]);
    float df_dWo = LHS_f * (u_f * Hw_o[i]);
    float di_dWo = LHS_i * (u_i * Hw_o[i]);
    float do_dWo = LHS_o * (x[i] + u_o * Hw_o[i]);
    Cw_o[i] = f * Cw_o[i] + old_c * df_dWo + i_val * dg_dWo + g * di_dWo;
    Hw_o[i] = o * (1 - tanh_of_c * tanh_of_c) * Cw_o[i] + tanh_of_c * do_dWo;
  }

  //  Section 1.6
  for (int i = 0; i < n; i++) {
    float dg_dWg = LHS_g * (x[i] + u_g * Hw_g[i]);
    float df_dWg = LHS_f * (u_f * Hw_g[i]);
    float di_dWg = LHS_i * (u_i * Hw_g[i]);
    float do_dWg = LHS_o * (u_o * Hw_g[i]);
    Cw_g[i] = f * Cw_g[i] + old_c * df_dWg + i_val * dg_dWg + g * di_dWg;
    Hw_g[i] = o * (1 - tanh_of_c * tanh_of_c) * Cw_g[i] + tanh_of_c * do_dWg;
  }

  //  Section 1.11
  {
    float dg_dbf = LHS_g * (u_g * Hb_f);
    float df_dbf = LHS_f * (1 + u_f * Hb_f);
    float di_dbf = LHS_i * (u_i * Hb_f);
    float do_dbf = LHS_o * (u_o * Hb_f);
    Cb_f = f * Cb_f + old_c * df_dbf + i_val * dg_dbf + g * di_dbf;
    Hb_f = o * (1 - tanh_of_c * tanh_of_c) * Cb_f + tanh_of_c * do_dbf;
  }

  //  Section 1.10
  {
    float dg_dbg = LHS_g * (1 + u_g * Hb_g);
    float df_dbg = LHS_f * (u_f * Hb_g);
    float di_dbg = LHS_i * (u_i * Hb_g);
    float do_dbg = LHS_o * (u_o * Hb_g);
    Cb_g = f * Cb_g + old_c * df_dbg + i_val * dg_dbg + g * di_dbg;
    Hb_g = o * (1 - tanh_of_c * tanh_of_c) * Cb_g + tanh_of_c * do_dbg;
  }

  // Section 1.12
  {
    float dg_dbo = LHS_g * (u_g * Hb_o);
    float df_dbo = LHS_f * (u_f * Hb_o);
    float di_dbo = LHS_i * (u_i * Hb_o);
    float do_dbo = LHS_o * (1 + u_o * Hb_o);
    Cb_o = f * Cb_o + old_c * df_dbo + i_val * dg_dbo + g * di_dbo;
    Hb_o = o * (1 - tanh_of_c * tanh_of_c) * Cb_o + tanh_of_c * do_dbo;
  }

  //  Section 1.3
  {
    float dg_dbi = LHS_g * (u_g * Hb_i);
    float df_dbi = LHS_f * (u_f * Hb_i);
    float di_dbi = LHS_i * (1 + u_i * Hb_i);
    float do_dbi = LHS_o * (u_o * Hb_i);
    Cb_i = f * Cb_i + old_c * df_dbi + i_val * dg_dbi + g * di_dbi;
    Hb_i = o * (1 - tanh_of_c * tanh_of_c) * Cb_i + tanh_of_c * do_dbi;
  }

  //  Section 1.8
  {
    float dg_duf = LHS_g * (u_g * Hu_f);
    float df_duf = LHS_f * (old_h + u_f * Hu_f);
    float di_duf = LHS_i * (u_i * Hu_f);
    float do_duf = LHS_o * (u_o * Hu_f);
    Cu_f = f * Cu_f + old_c * df_duf + i_val * dg_duf + g * di_duf;
    Hu_f = o * (1 - tanh_of_c * tanh_of_c) * Cu_f + tanh_of_c * do_duf;
  }

  //  Section 1.7
  {
    float dg_duo = LHS_g * (u_g * Hu_o);
    float df_duo = LHS_f * (u_f * Hu_o);
    float di_duo = LHS_i * (u_i * Hu_o);
    float do_duo = LHS_o * (old_h + u_o * Hu_o);
    Cu_o = f * Cu_o + old_c * df_duo + i_val * dg_duo + g * di_duo;
    Hu_o = o * (1 - tanh_of_c * tanh_of_c) * Cu_o + tanh_of_c * do_duo;
  }

  //  Section 1.9
  {
    float dg_dug = LHS_g * (old_h + u_g * Hu_g);
    float df_dug = LHS_f * (u_f * Hu_g);
    float di_dug = LHS_i * (u_i * Hu_g);
    float do_dug = LHS_o * (u_o * Hu_g);
    Cu_g = f * Cu_g + old_c * df_dug + i_val * dg_dug + g * di_dug;
    Hu_g = o * (1 - tanh_of_c * tanh_of_c) * Cu_g + tanh_of_c * do_dug;
  }

  //  Section 1.2
  {
    float dg_dui = LHS_g * (u_g * Hu_i);
    float df_dui = LHS_f * (u_f * Hu_i);
    float di_dui = LHS_i * (old_h + u_i * Hu_i);
    float do_dui = LHS_o * (u_o * Hu_i);
    Cu_i = f * Cu_i + old_c * df_dui + i_val * dg_dui + g * di_dui;
    Hu_i = o * (1 - tanh_of_c * tanh_of_c) * Cu_i + tanh_of_c * do_dui;
  }
  update_statistics();
}

int LSTM::get_users() { return users; }

void LSTM::increment_user() { users++; }

void LSTM::decrement_user() { users--; }

void LSTM::zero_grad() {

  Gb_i = Gb_f = Gb_g = Gb_o = 0;
  Gu_i = Gu_f = Gu_g = Gu_o = 0;

  for (int counter = 0; counter < Hw_f.size(); counter++) {

    this->Gw_i[counter] = 0;
    this->Gw_f[counter] = 0;
    this->Gw_g[counter] = 0;
    this->Gw_o[counter] = 0;
  }
}

void LSTM::reset_state() {
  Hu_i = Hu_f = Hu_g = Hu_o = 0;
  Cu_i = Cu_f = Cu_g = Cu_o = 0;
  Hb_i = Hb_f = Hb_g = Hb_o = 0;
  Cb_i = Cb_f = Cb_g = Cb_o = 0;

  for (int counter = 0; counter < Hw_f.size(); counter++) {
    this->Hw_i[counter] = 0;
    this->Hw_f[counter] = 0;
    this->Hw_g[counter] = 0;
    this->Hw_o[counter] = 0;

    this->Cw_i[counter] = 0;
    this->Cw_f[counter] = 0;
    this->Cw_g[counter] = 0;
    this->Cw_o[counter] = 0;
  }

  h = 0;
  c = 0;
  this->value = 0;
}

void LSTM::accumulate_gradient(float incoming_grad) {
  int n = this->w_f.size();
  for (int counter = 0; counter < n; counter++) {
    Gw_f[counter] += Hw_f[counter] * incoming_grad;
    Gw_i[counter] += Hw_i[counter] * incoming_grad;
    Gw_o[counter] += Hw_o[counter] * incoming_grad;
    Gw_g[counter] += Hw_g[counter] * incoming_grad;
  }
  Gb_f += Hb_f * incoming_grad;
  Gb_i += Hb_i * incoming_grad;
  Gb_o += Hb_o * incoming_grad;
  Gb_g += Hb_g * incoming_grad;
  Gu_f += Hu_f * incoming_grad;
  Gu_i += Hu_i * incoming_grad;
  Gu_o += Hu_o * incoming_grad;
  Gu_g += Hu_g * incoming_grad;
};

void LSTM::decay_gradient(float decay_rate) {
  int n = this->w_f.size();
  for (int counter = 0; counter < n; counter++) {
    Gw_f[counter] *= decay_rate;
    Gw_i[counter] *= decay_rate;
    Gw_o[counter] *= decay_rate;
    Gw_g[counter] *= decay_rate;
  }
  Gb_f *= decay_rate;
  Gb_i *= decay_rate;
  Gb_o *= decay_rate;
  Gb_g *= decay_rate;
  Gu_f *= decay_rate;
  Gu_i *= decay_rate;
  Gu_o *= decay_rate;
  Gu_g *= decay_rate;
}

void LSTM::update_weights(float step_size, float error) {
  int n = this->w_f.size();
//  step_size /= n;
  float b2_rate = 0.9999;
  for (int counter = 0; counter < n; counter++) {
    beta_2_w_f[counter] =
        b2_rate * beta_2_w_f[counter] + (1 - b2_rate) * (Gw_f[counter] * error)*(Gw_f[counter] * error);
    beta_2_w_i[counter] =
        b2_rate * beta_2_w_i[counter] + (1 - b2_rate) * (Gw_i[counter] * error)*(Gw_i[counter] * error);
    beta_2_w_o[counter] =
        b2_rate * beta_2_w_o[counter] + (1 - b2_rate) * (Gw_o[counter] * error)*(Gw_o[counter] * error);
    beta_2_w_g[counter] =
        b2_rate * beta_2_w_g[counter] + (1 - b2_rate) * (Gw_g[counter] * error)*(Gw_g[counter] * error);
  }
  for (int counter = 0; counter < n; counter++) {
    w_f[counter] += (step_size / (sqrt(beta_2_w_f[counter]) + epsilon)) * Gw_f[counter] * error;
    w_i[counter] += (step_size / (sqrt(beta_2_w_i[counter]) + epsilon)) * Gw_i[counter] * error;
    w_o[counter] += (step_size / (sqrt(beta_2_w_o[counter]) + epsilon)) * Gw_o[counter] * error;
    w_g[counter] += (step_size / (sqrt(beta_2_w_g[counter]) + epsilon)) * Gw_g[counter] * error;
  }


  beta_2_b_f = b2_rate*beta_2_b_f + (1-b2_rate) * (Gb_f  * error)*(Gb_f  * error);
  beta_2_b_i = b2_rate*beta_2_b_i + (1-b2_rate) * (Gb_i  * error)*(Gb_i  * error);
  beta_2_b_o = b2_rate*beta_2_b_o + (1-b2_rate) * (Gb_o  * error)*(Gb_o  * error);
  beta_2_b_g = b2_rate*beta_2_b_g + (1-b2_rate) * (Gb_g  * error)*(Gb_g  * error);
  beta_2_u_f = b2_rate*beta_2_u_f + (1-b2_rate) * (Gu_f  * error)*(Gu_f  * error);
  beta_2_u_i = b2_rate*beta_2_u_i + (1-b2_rate) * (Gu_i  * error)*(Gu_i  * error);
  beta_2_u_o = b2_rate*beta_2_u_o + (1-b2_rate) * (Gu_o  * error)*(Gu_o  * error);
  beta_2_u_g = b2_rate*beta_2_u_g + (1-b2_rate) * (Gu_g  * error)*(Gu_g  * error);

  b_f += (step_size/ (sqrt(beta_2_b_f) + epsilon)) * Gb_f  * error;
  b_i += (step_size/ (sqrt(beta_2_b_i) + epsilon)) * Gb_i  * error;
  b_o += (step_size/ (sqrt(beta_2_b_o) + epsilon)) * Gb_o  * error;
  b_g += (step_size/ (sqrt(beta_2_b_g) + epsilon)) * Gb_g  * error;
  u_f += (step_size/ (sqrt(beta_2_u_f) + epsilon)) * Gu_f  * error;
  u_i += (step_size/ (sqrt(beta_2_u_i) + epsilon)) * Gu_i  * error;
  u_o += (step_size/ (sqrt(beta_2_u_o) + epsilon)) * Gu_o  * error;
  u_g += (step_size/ (sqrt(beta_2_u_g) + epsilon)) * Gu_g  * error;
}
void LSTM::update_weights(float step_size) {
  int n = this->w_f.size();
  for (int counter = 0; counter < n; counter++) {
    w_f[counter] += Gw_f[counter] * step_size;
    w_i[counter] += Gw_i[counter] * step_size;
    w_o[counter] += Gw_o[counter] * step_size;
    w_g[counter] += Gw_g[counter] * step_size;
  }
  b_f += Gb_f * step_size;
  b_i += Gb_i * step_size;
  b_o += Gb_o * step_size;
  b_g += Gb_g * step_size;
  u_f += Gu_f * step_size;
  u_i += Gu_i * step_size;
  u_o += Gu_o * step_size;
  u_g += Gu_g * step_size;
}

float LSTM::get_value_without_sideeffects() {

  float i_val_t, g_t, f_t, o_t;
  float old_h_t, old_c_t;

  old_h_t = h;
  old_c_t = c;

  i_val_t = b_i;
  g_t = b_g;
  f_t = b_f;
  o_t = b_o;

  auto x = get_normalized_values();
  for (int counter_t = 0; counter_t < x.size(); counter_t++) {
    i_val_t += this->w_i[counter_t] * x[counter_t];
    f_t += this->w_f[counter_t] * x[counter_t];
    g_t += this->w_g[counter_t] * x[counter_t];
    o_t += this->w_o[counter_t] * x[counter_t];
  }

  //  Self connection
  i_val_t += u_i * old_h_t;
  g_t += u_g * old_h_t;
  f_t += u_f * old_h_t;
  o_t += u_o * old_h_t;

  //  Applying non-linear transformations and updating the hidden state;
  i_val_t = sigmoid(i_val_t);
  f_t = sigmoid(f_t);
  o_t = sigmoid(o_t);
  g_t = tanh(g_t);

  float c_t = f_t * old_c_t + i_val_t * g_t;
  float h_t = o_t * tanh(c_t);
  //  if (h_t > 1)
  //    h_t = 1;
  //  if (h_t < -1)
  //    h_t = -1;
  return h_t;
}

void LSTM::update_value_sync() {
  this->value_before_firing = 0;
  old_h = h;
  old_c = c;
  //  Adding bias value
  i_val = b_i;
  g = b_g;
  f = b_f;
  o = b_o;

  auto x = get_normalized_values();
  for (int counter = 0; counter < x.size(); counter++) {
    i_val += this->w_i[counter] * x[counter];
    f += this->w_f[counter] * x[counter];
    g += this->w_g[counter] * x[counter];
    o += this->w_o[counter] * x[counter];
  }

  //  Self connection
  i_val += u_i * old_h;
  g += u_g * old_h;
  f += u_f * old_h;
  o += u_o * old_h;

  //  Applying non-linear transformations and updating the hidden state;
  i_val = sigmoid(i_val);
  f = sigmoid(f);
  o = sigmoid(o);
  g = tanh(g);

  c = f * old_c + i_val * g;
  h = o * tanh(c);
}

float LSTM::get_hidden_state() { return this->h; }

std::vector<float> LSTM::get_normalized_values() {
  std::vector<float> values_ret;
  values_ret.reserve(this->incoming_neurons.size());
  for (int counter = 0; counter < this->incoming_neurons.size(); counter++) {
    values_ret.push_back(
        (this->incoming_neurons[counter]->value - this->input_means[counter]) /
        sqrt(input_std[counter]));
  }
  return values_ret;
}
void LSTM::update_statistics() {
  if (update_statistics_flag) {
    for (int counter = 0; counter < this->incoming_neurons.size(); counter++) {
      float val = this->incoming_neurons[counter]->value;
      this->input_means[counter] =
          this->input_means[counter] * this->decay_rate +
          (1 - decay_rate) * val;
      this->input_std[counter] = this->input_std[counter] * decay_rate +
                                 (1 - decay_rate) *
                                     (val - this->input_means[counter]) *
                                     (val - this->input_means[counter]);
      if (this->input_std[counter] < this->std_cap)
        this->input_std[counter] = this->std_cap;
    }
  }
}

void LSTM::add_synapse(Neuron *s, float w_i, float w_f, float w_g, float w_o) {
  this->incoming_neurons.push_back(s);
  this->w_i.push_back(w_i);
  this->w_f.push_back(w_f);
  this->w_g.push_back(w_g);
  this->w_o.push_back(w_o);
  this->input_means.push_back(0);
  this->input_std.push_back(1);

  this->Hw_i.push_back(0);
  this->Hw_f.push_back(0);
  this->Hw_g.push_back(0);
  this->Hw_o.push_back(0);

  this->Cw_i.push_back(0);
  this->Cw_f.push_back(0);
  this->Cw_g.push_back(0);
  this->Cw_o.push_back(0);

  this->Gw_i.push_back(0);
  this->Gw_f.push_back(0);
  this->Gw_g.push_back(0);
  this->Gw_o.push_back(0);

  this->beta_2_w_i.push_back(1);
  this->beta_2_w_f.push_back(1);
  this->beta_2_w_g.push_back(1);
  this->beta_2_w_o.push_back(1);
}

LSTMNormalied::LSTMNormalied(float ui, float uf, float ug, float uo, float bi,
                             float bf, float bg, float bo, float std_cap,
                             float decay_rate)
    : LSTM(ui, uf, ug, uo, bi, bf, bg, bo, std_cap, decay_rate) {
  this->state_mean = 0;
  this->state_variance = 1;
}

void LSTMNormalied::update_statistics() {
  LSTM::update_statistics();
  float temp = this->decay_rate;
  //  this->decay_rate = 0.99;
  this->state_mean = this->state_mean * this->decay_rate +
                     (1 - this->decay_rate) * this->value;
  float val_minus_mean = this->value - this->state_mean;
  this->state_variance =
      this->state_variance * this->decay_rate +
      (1 - this->decay_rate) * val_minus_mean * val_minus_mean;
  if (this->state_variance < this->std_cap)
    this->state_variance = this->std_cap;
  //  this->decay_rate = temp;
  //  std::cout << "Mean " << this->state_mean << " " << this->state_variance <<
  //  std::endl; this->state_mean = -2; this->state_variance = 9;
}

void LSTMNormalied::update_value_sync() {
  this->value_before_firing = 0;
  old_h = h;
  old_c = c;
  //  Adding bias value
  i_val = b_i;
  g = b_g;
  f = b_f;
  o = b_o;

  auto x = get_normalized_values();
  for (int counter = 0; counter < x.size(); counter++) {

    i_val += this->w_i[counter] * x[counter];
    f += this->w_f[counter] * x[counter];
    g += this->w_g[counter] * x[counter];
    o += this->w_o[counter] * x[counter];
  }

  //  Self connection
  float normalized_old_h = this->get_normalized_hidden_state(old_h);
  i_val += u_i * normalized_old_h;
  g += u_g * normalized_old_h;
  f += u_f * normalized_old_h;
  o += u_o * normalized_old_h;

  //  Applying non-linear transformations and updating the hidden state;
  i_val = sigmoid(i_val);
  f = sigmoid(f);
  o = sigmoid(o);
  g = tanh(g);

  c = f * old_c + i_val * g;
  h = o * tanh(c);
}

float LSTMNormalied::get_normalized_hidden_state(float h) {
  return (h - this->state_mean) / (sqrt(state_variance));
}
void LSTMNormalied::compute_gradient_of_all_synapses() {
  std::vector<float> x = get_normalized_values();
  int n = this->incoming_neurons.size();

  //  Computing terms that are reused to save flops

  float LHS_g = (1 - (g * g));
  float LHS_f = f * (1 - f);
  float LHS_i = i_val * (1 - i_val);
  float LHS_o = o * (1 - o);
  float tanh_of_c = tanh(c);

  //  Section 1.1

  for (int i = 0; i < n; i++) {
    float dg_dWi = LHS_g * (u_g * (Hw_i[i] / sqrt(state_variance)));
    float df_dWi = LHS_f * (u_f * (Hw_i[i] / sqrt(state_variance)));
    float di_dWi = LHS_i * (x[i] + u_i * (Hw_i[i] / sqrt(state_variance)));
    float do_dWi = LHS_o * (u_o * (Hw_i[i] / sqrt(state_variance)));
    Cw_i[i] = f * Cw_i[i] + old_c * df_dWi + i_val * dg_dWi + g * di_dWi;
    Hw_i[i] = o * (1 - tanh_of_c * tanh_of_c) * Cw_i[i] + tanh_of_c * do_dWi;
  }

  //  Section 1.4
  for (int i = 0; i < n; i++) {
    float dg_dWF = LHS_g * (u_g * (Hw_f[i] / sqrt(state_variance)));
    float df_dWF = LHS_f * (x[i] + u_f * (Hw_f[i] / sqrt(state_variance)));
    float di_dWF = LHS_i * (u_i * (Hw_f[i] / sqrt(state_variance)));
    float do_dWF = LHS_o * (u_o * (Hw_f[i] / sqrt(state_variance)));
    Cw_f[i] = f * Cw_f[i] + old_c * df_dWF + i_val * dg_dWF + g * di_dWF;
    Hw_f[i] = o * (1 - tanh_of_c * tanh_of_c) * Cw_f[i] + tanh_of_c * do_dWF;
  }

  //  Section 1.5
  for (int i = 0; i < n; i++) {
    float dg_dWo = LHS_g * (u_g * (Hw_o[i] / sqrt(state_variance)));
    float df_dWo = LHS_f * (u_f * (Hw_o[i] / sqrt(state_variance)));
    float di_dWo = LHS_i * (u_i * (Hw_o[i] / sqrt(state_variance)));
    float do_dWo = LHS_o * (x[i] + u_o * (Hw_o[i] / sqrt(state_variance)));
    Cw_o[i] = f * Cw_o[i] + old_c * df_dWo + i_val * dg_dWo + g * di_dWo;
    Hw_o[i] = o * (1 - tanh_of_c * tanh_of_c) * Cw_o[i] + tanh_of_c * do_dWo;
  }

  //  Section 1.6
  for (int i = 0; i < n; i++) {
    float dg_dWg = LHS_g * (x[i] + u_g * (Hw_g[i] / sqrt(state_variance)));
    float df_dWg = LHS_f * (u_f * (Hw_g[i] / sqrt(state_variance)));
    float di_dWg = LHS_i * (u_i * (Hw_g[i] / sqrt(state_variance)));
    float do_dWg = LHS_o * (u_o * (Hw_g[i] / sqrt(state_variance)));
    Cw_g[i] = f * Cw_g[i] + old_c * df_dWg + i_val * dg_dWg + g * di_dWg;
    Hw_g[i] = o * (1 - tanh_of_c * tanh_of_c) * Cw_g[i] + tanh_of_c * do_dWg;
  }

  //  Section 1.11
  {
    float dg_dbf = LHS_g * (u_g * (Hb_f / sqrt(state_variance)));
    float df_dbf = LHS_f * (1 + u_f * (Hb_f / sqrt(state_variance)));
    float di_dbf = LHS_i * (u_i * (Hb_f / sqrt(state_variance)));
    float do_dbf = LHS_o * (u_o * (Hb_f / sqrt(state_variance)));
    Cb_f = f * Cb_f + old_c * df_dbf + i_val * dg_dbf + g * di_dbf;
    Hb_f = o * (1 - tanh_of_c * tanh_of_c) * Cb_f + tanh_of_c * do_dbf;
  }

  //  Section 1.10
  {
    float dg_dbg = LHS_g * (1 + u_g * (Hb_g / sqrt(state_variance)));
    float df_dbg = LHS_f * (u_f * (Hb_g / sqrt(state_variance)));
    float di_dbg = LHS_i * (u_i * (Hb_g / sqrt(state_variance)));
    float do_dbg = LHS_o * (u_o * (Hb_g / sqrt(state_variance)));
    Cb_g = f * Cb_g + old_c * df_dbg + i_val * dg_dbg + g * di_dbg;
    Hb_g = o * (1 - tanh_of_c * tanh_of_c) * Cb_g + tanh_of_c * do_dbg;
  }

  // Section 1.12
  {
    float dg_dbo = LHS_g * (u_g * (Hb_o / sqrt(state_variance)));
    float df_dbo = LHS_f * (u_f * (Hb_o / sqrt(state_variance)));
    float di_dbo = LHS_i * (u_i * (Hb_o / sqrt(state_variance)));
    float do_dbo = LHS_o * (1 + u_o * (Hb_o / sqrt(state_variance)));
    Cb_o = f * Cb_o + old_c * df_dbo + i_val * dg_dbo + g * di_dbo;
    Hb_o = o * (1 - tanh_of_c * tanh_of_c) * Cb_o + tanh_of_c * do_dbo;
  }

  //  Section 1.3
  {
    float dg_dbi = LHS_g * (u_g * (Hb_i / sqrt(state_variance)));
    float df_dbi = LHS_f * (u_f * (Hb_i / sqrt(state_variance)));
    float di_dbi = LHS_i * (1 + u_i * (Hb_i / sqrt(state_variance)));
    float do_dbi = LHS_o * (u_o * (Hb_i / sqrt(state_variance)));
    Cb_i = f * Cb_i + old_c * df_dbi + i_val * dg_dbi + g * di_dbi;
    Hb_i = o * (1 - tanh_of_c * tanh_of_c) * Cb_i + tanh_of_c * do_dbi;
  }

  //  Section 1.8
  {
    float dg_duf = LHS_g * (u_g * (Hu_f / sqrt(state_variance)));
    float df_duf = LHS_f * (get_normalized_hidden_state(old_h) +
                            u_f * (Hu_f / sqrt(state_variance)));
    float di_duf = LHS_i * (u_i * (Hu_f / sqrt(state_variance)));
    float do_duf = LHS_o * (u_o * (Hu_f / sqrt(state_variance)));
    Cu_f = f * Cu_f + old_c * df_duf + i_val * dg_duf + g * di_duf;
    Hu_f = o * (1 - tanh_of_c * tanh_of_c) * Cu_f + tanh_of_c * do_duf;
  }

  //  Section 1.7
  {
    float dg_duo = LHS_g * (u_g * (Hu_o / sqrt(state_variance)));
    float df_duo = LHS_f * (u_f * (Hu_o / sqrt(state_variance)));
    float di_duo = LHS_i * (u_i * (Hu_o / sqrt(state_variance)));
    float do_duo = LHS_o * (get_normalized_hidden_state(old_h) +
                            u_o * (Hu_o / sqrt(state_variance)));
    Cu_o = f * Cu_o + old_c * df_duo + i_val * dg_duo + g * di_duo;
    Hu_o = o * (1 - tanh_of_c * tanh_of_c) * Cu_o + tanh_of_c * do_duo;
  }

  //  Section 1.9
  {
    float dg_dug = LHS_g * (get_normalized_hidden_state(old_h) +
                            u_g * (Hu_g / sqrt(state_variance)));
    float df_dug = LHS_f * (u_f * (Hu_g / sqrt(state_variance)));
    float di_dug = LHS_i * (u_i * (Hu_g / sqrt(state_variance)));
    float do_dug = LHS_o * (u_o * (Hu_g / sqrt(state_variance)));
    Cu_g = f * Cu_g + old_c * df_dug + i_val * dg_dug + g * di_dug;
    Hu_g = o * (1 - tanh_of_c * tanh_of_c) * Cu_g + tanh_of_c * do_dug;
  }

  //  Section 1.2
  {
    float dg_dui = LHS_g * (u_g * (Hu_i / sqrt(state_variance)));
    float df_dui = LHS_f * (u_f * (Hu_i / sqrt(state_variance)));
    float di_dui = LHS_i * (get_normalized_hidden_state(old_h) +
                            u_i * (Hu_i / sqrt(state_variance)));
    float do_dui = LHS_o * (u_o * (Hu_i / sqrt(state_variance)));
    Cu_i = f * Cu_i + old_c * df_dui + i_val * dg_dui + g * di_dui;
    Hu_i = o * (1 - tanh_of_c * tanh_of_c) * Cu_i + tanh_of_c * do_dui;
  }
  update_statistics();
}

float LSTMNormalied::get_value_without_sideeffects() {

  float i_val_t, g_t, f_t, o_t;
  float old_h_t, old_c_t;

  old_h_t = get_normalized_hidden_state(h);
  old_c_t = c;

  i_val_t = b_i;
  g_t = b_g;
  f_t = b_f;
  o_t = b_o;

  auto x = get_normalized_values();
  for (int counter_t = 0; counter_t < x.size(); counter_t++) {
    i_val_t += this->w_i[counter_t] * x[counter_t];
    f_t += this->w_f[counter_t] * x[counter_t];
    g_t += this->w_g[counter_t] * x[counter_t];
    o_t += this->w_o[counter_t] * x[counter_t];
  }

  //  Self connection
  i_val_t += u_i * old_h_t;
  g_t += u_g * old_h_t;
  f_t += u_f * old_h_t;
  o_t += u_o * old_h_t;

  //  Applying non-linear transformations and updating the hidden state;
  i_val_t = sigmoid(i_val_t);
  f_t = sigmoid(f_t);
  o_t = sigmoid(o_t);
  g_t = tanh(g_t);

  float c_t = f_t * old_c_t + i_val_t * g_t;
  float h_t = o_t * tanh(c_t);
  return h_t;
}

void LSTMNormalied::fire() {

  this->value = h;
  this->neuron_age++;
}