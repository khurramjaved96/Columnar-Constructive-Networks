//
// Created by Khurram Javed on 2022-02-23.
//


/* Efficient single core implementation of fully connected LSTMs and truncated-BPTT
 * The predictions, hidden states, and gradients were compared to an existing autograd---PyTorch---is all of them match PyTorch's implementation of LSTMs.
 * One difference between the PyTorch LSTM implementation, and this implementation is that I only use a single bias term per gate. Using two bias terms
 * is redundant.
 * For small networks, this implementation can be 50 - 100x faster than PyTorch or Libtorch.
 * */


#include "../../../include/nn/networks/lstm_bptt.h"
#include "../../../include/utils.h"
#include "../../../include/nn/utils.h"
#include <random>

void DenseLSTM::print_features_stats() {}

DenseLSTMRmsProp::DenseLSTMRmsProp(float step_size,
                                   int seed,
                                   int hidden_size,
                                   int no_of_input_features,
                                   int truncation, float beta_2, float epsilon) : DenseLSTM(step_size,
                                                                                            seed,
                                                                                            hidden_size,
                                                                                            no_of_input_features,
                                                                                            truncation) {
  this->beta_2 = beta_2;
  this->epsilon = epsilon;
  for (int outer_counter = 0; outer_counter < hidden_size; outer_counter++) {
    for (int inner_counter = 0; inner_counter < input_size * 4; inner_counter++) {
      this->W_grad_rmsprop.push_back(1);
    }
    for (int outer_counter = 0; outer_counter < hidden_size; outer_counter++) {
      for (int inner_counter = 0; inner_counter < hidden_size * 4; inner_counter++) {
        this->U_grad_rmsprop.push_back(1);
      }
    }
    for (int outer_counter = 0; outer_counter < hidden_size * 4; outer_counter++) {
      this->b_grad_rmsprop.push_back(1);
    }
    for (int counter = 0; counter < hidden_size; counter++) {
      this->prediction_weights_grad_rmsprop.push_back(1);
    }
  }
}

void DenseLSTMRmsProp::update_parameters(int layer, float error) {
//  std::cout << "Being called\n";
  for (int counter = 0; counter < this->prediction_weights_grad.size(); counter++) {
    prediction_weights_grad_rmsprop[counter] = prediction_weights_grad_rmsprop[counter] * beta_2
        + (1 - beta_2) * (this->prediction_weights_grad[counter] * error) * (this->prediction_weights_grad[counter] * error);
    this->prediction_weights[counter] +=
        (step_size * error  * this->prediction_weights_grad[counter])
            / (sqrt(prediction_weights_grad_rmsprop[counter]) + this->epsilon);
  }
  for (int counter = 0; counter < W_grad.size(); counter++) {
    W_grad_rmsprop[counter] =
        W_grad_rmsprop[counter] * beta_2 + (1 - beta_2) * (W_grad[counter] * error) * (W_grad[counter] * error);
    W[counter] += step_size * error * W_grad[counter] / (sqrt(W_grad_rmsprop[counter]) + this->epsilon);
  }
  for (int counter = 0; counter < U_grad.size(); counter++) {
    U_grad_rmsprop[counter] =
        U_grad_rmsprop[counter] * beta_2 + (1 - beta_2) * (U_grad[counter] * error) * (U_grad[counter] * error);
    U[counter] += step_size * error * U_grad[counter] / (sqrt(U_grad_rmsprop[counter]) + this->epsilon);
  }
  for (int counter = 0; counter < b_grad.size(); counter++) {
    b_grad_rmsprop[counter] =
        b_grad_rmsprop[counter] * beta_2 + (1 - beta_2) * (b_grad[counter] * error) * (b_grad[counter] * error);
    b[counter] += step_size * error * b_grad[counter] / (sqrt(b_grad_rmsprop[counter]) + this->epsilon);
  }

//  for(int c = 0; c < 2; c++) {
//    std::cout << "C = " << c << std::endl;
//    std::cout << "Prediction grad " << error * prediction_weights_grad[c] << std::endl;
//    std::cout << "Normalizing val = " << (sqrt(prediction_weights_grad_rmsprop[c])) << std::endl;
////  std::cout << "Grad = " << (error * prediction_weights_grad[0])*(error * prediction_weights_grad[0]) << std::endl;
//    std::cout << "Normalized grad = "
//              << (error * prediction_weights_grad[c]) / (sqrt(prediction_weights_grad_rmsprop[c]) + this->epsilon)
//              << std::endl;
//  }
}
//
DenseLSTM::DenseLSTM(float step_size,
                     int seed,
                     int hidden_size,
                     int no_of_input_features,
                     int truncation) : mt(seed) {
  this->time_step = 1;
  this->step_size = step_size;
  this->input_size = no_of_input_features;
  this->hidden_state_size = hidden_size;
  this->truncation = truncation;
  int t = time_step % truncation;
  for (int counter = 0; counter < truncation; counter++) {
    std::vector<float> f_temp(hidden_size, 0), g_temp(hidden_size, 0), o_temp(hidden_size, 0), i_temp(hidden_size, 0),
        c_temp(hidden_size, 0), h_temp(hidden_size, 0);
    std::vector<float> x_temp(input_size, 0);
    x_queue.push_back(x_temp);
    f_queue.push_back(f_temp);
    g_queue.push_back(g_temp);
    o_queue.push_back(o_temp);
    i_queue.push_back(i_temp);
    c_queue.push_back(c_temp);
    h_queue.push_back(h_temp);
  }
  for (int counter = 0; counter < hidden_size; counter++) {
    this->prediction_weights.push_back(0);
    this->prediction_weights_grad.push_back(0);
  }

  std::uniform_real_distribution<float> weight_sampler(-sqrt(1.0 / float(hidden_size)), sqrt(1.0 / float(hidden_size)));
  for (int outer_counter = 0; outer_counter < hidden_size; outer_counter++) {
    std::vector<float> temp_weight_vector;
    for (int inner_counter = 0; inner_counter < input_size * 4; inner_counter++) {
      this->W.push_back(weight_sampler(this->mt));
      this->W_grad.push_back(0);
    }
  }
  for (int outer_counter = 0; outer_counter < hidden_size; outer_counter++) {
    std::vector<float> temp_weight_vector;
    for (int inner_counter = 0; inner_counter < hidden_size * 4; inner_counter++) {
      this->U.push_back(weight_sampler(this->mt));
      this->U_grad.push_back(0);
    }
  }

  for (int outer_counter = 0; outer_counter < hidden_size * 4; outer_counter++) {
    this->b.push_back(weight_sampler(this->mt));
    this->b_grad.push_back(0);
  }

}

float DenseLSTM::get_target_without_sideeffects(std::vector<float> inputs) {
  //  f->g->i->o
//  compute f, g, i and o for each recurrent neuron
  std::vector<float> f_temp, g_temp, i_temp, o_temp;
  f_temp.reserve(this->hidden_state_size);
  g_temp.reserve(this->hidden_state_size);
  i_temp.reserve(this->hidden_state_size);
  o_temp.reserve(this->hidden_state_size);
  int t = this->time_step % truncation;
  int t_1 = (this->time_step - 1) % truncation;
  if (t_1 < 0)
    t_1 = truncation + t_1;

  float f_temp_cur, g_temp_cur, i_temp_cur, o_temp_cur;
  f_temp_cur = g_temp_cur = i_temp_cur = o_temp_cur = 0;


// compute i
  for (int h_index = 0; h_index < this->hidden_state_size; h_index++) {
    i_temp_cur = 0;
    for (int inp_index = 0; inp_index < this->input_size; inp_index++) {
      i_temp_cur += W[0 * this->input_size * hidden_state_size + h_index * input_size + inp_index] * inputs[inp_index];
//      std::cout << "W_i weights " << "h_index " << h_index << " inp index " << inp_index << " " << W[0 * this->input_size*hidden_state_size + h_index*input_size + inp_index] << std::endl;
    }
    for (int old_h_index = 0; old_h_index < this->hidden_state_size; old_h_index++) {
      i_temp_cur += U[0 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + old_h_index]
          * h_queue[t_1][old_h_index];
    }
    i_temp_cur += b[0 * hidden_state_size + h_index];
    i_temp_cur = sigmoid(i_temp_cur);
    i_temp.push_back(i_temp_cur);
  }

// compute f
  for (int h_index = 0; h_index < this->hidden_state_size; h_index++) {
    f_temp_cur = 0;
    for (int inp_index = 0; inp_index < this->input_size; inp_index++) {
      f_temp_cur += W[1 * this->input_size * hidden_state_size + h_index * input_size + inp_index] * inputs[inp_index];
    }
    for (int old_h_index = 0; old_h_index < this->hidden_state_size; old_h_index++) {
      f_temp_cur += U[1 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + old_h_index]
          * h_queue[t_1][old_h_index];
    }

    f_temp_cur += b[1 * hidden_state_size + h_index];
    f_temp_cur = sigmoid(f_temp_cur);
    f_temp.push_back(f_temp_cur);
  }

//  Compute g
  for (int h_index = 0; h_index < this->hidden_state_size; h_index++) {
    g_temp_cur = 0;
    for (int inp_index = 0; inp_index < this->input_size; inp_index++) {
      g_temp_cur += W[2 * this->input_size * hidden_state_size + h_index * input_size + inp_index] * inputs[inp_index];
    }
    for (int old_h_index = 0; old_h_index < this->hidden_state_size; old_h_index++) {
      g_temp_cur += U[2 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + old_h_index]
          * h_queue[t_1][old_h_index];
    }
    g_temp_cur += b[2 * hidden_state_size + h_index];
    g_temp_cur = tanh(g_temp_cur);
    g_temp.push_back(g_temp_cur);
  }


//  Compute o
  for (int h_index = 0; h_index < this->hidden_state_size; h_index++) {
    o_temp_cur = 0;
    for (int inp_index = 0; inp_index < this->input_size; inp_index++) {
//      std::cout << "O weigjt = " << W[h_index][3 * this->input_size + inp_index] << std::endl;
      o_temp_cur += W[3 * this->input_size * hidden_state_size + h_index * input_size + inp_index] * inputs[inp_index];
    }
    for (int old_h_index = 0; old_h_index < this->hidden_state_size; old_h_index++) {
      o_temp_cur += U[3 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + old_h_index]
          * h_queue[t_1][old_h_index];
    }
    o_temp_cur += b[3 * hidden_state_size + h_index];
    o_temp_cur = sigmoid(o_temp_cur);
    o_temp.push_back(o_temp_cur);
  }

  std::vector<float> c_temp(hidden_state_size, 0);
  std::vector<float> h_temp(hidden_state_size, 0);
  for (int h_index = 0; h_index < this->hidden_state_size; h_index++) {
    c_temp[h_index] = f_temp[h_index] * c_queue[t_1][h_index] + i_temp[h_index] * g_temp[h_index];
    h_temp[h_index] = o_temp[h_index] * tanh(c_temp[h_index]);
  }

  float prediction_temp = 0;
  for (int counter = 0; counter < prediction_weights.size(); counter++) {
    prediction_temp += h_temp[counter] * prediction_weights[counter];
  }
  return prediction_temp;
}

float DenseLSTM::forward(std::vector<float> inputs) {
//  f->g->i->o
//  compute f, g, i and o for each recurrent neuron
  std::vector<float> f_temp, g_temp, i_temp, o_temp;
  f_temp.reserve(this->hidden_state_size);
  g_temp.reserve(this->hidden_state_size);
  i_temp.reserve(this->hidden_state_size);
  o_temp.reserve(this->hidden_state_size);
  int t = this->time_step % truncation;
  int t_1 = (this->time_step - 1) % truncation;
  if (t_1 < 0)
    t_1 = truncation + t_1;

  float f_temp_cur, g_temp_cur, i_temp_cur, o_temp_cur;
  f_temp_cur = g_temp_cur = i_temp_cur = o_temp_cur = 0;


// compute i
  for (int h_index = 0; h_index < this->hidden_state_size; h_index++) {
    i_temp_cur = 0;
    for (int inp_index = 0; inp_index < this->input_size; inp_index++) {
      i_temp_cur += W[0 * this->input_size * hidden_state_size + h_index * input_size + inp_index] * inputs[inp_index];
//      std::cout << "W_i weights " << "h_index " << h_index << " inp index " << inp_index << " " << W[0 * this->input_size*hidden_state_size + h_index*input_size + inp_index] << std::endl;
    }
    for (int old_h_index = 0; old_h_index < this->hidden_state_size; old_h_index++) {
      i_temp_cur += U[0 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + old_h_index]
          * h_queue[t_1][old_h_index];
    }
    i_temp_cur += b[0 * hidden_state_size + h_index];
    i_temp_cur = sigmoid(i_temp_cur);
    i_temp.push_back(i_temp_cur);
  }

// compute f
  for (int h_index = 0; h_index < this->hidden_state_size; h_index++) {
    f_temp_cur = 0;
    for (int inp_index = 0; inp_index < this->input_size; inp_index++) {
      f_temp_cur += W[1 * this->input_size * hidden_state_size + h_index * input_size + inp_index] * inputs[inp_index];
    }
    for (int old_h_index = 0; old_h_index < this->hidden_state_size; old_h_index++) {
      f_temp_cur += U[1 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + old_h_index]
          * h_queue[t_1][old_h_index];
    }

    f_temp_cur += b[1 * hidden_state_size + h_index];
    f_temp_cur = sigmoid(f_temp_cur);
    f_temp.push_back(f_temp_cur);
  }

//  Compute g
  for (int h_index = 0; h_index < this->hidden_state_size; h_index++) {
    g_temp_cur = 0;
    for (int inp_index = 0; inp_index < this->input_size; inp_index++) {
      g_temp_cur += W[2 * this->input_size * hidden_state_size + h_index * input_size + inp_index] * inputs[inp_index];
    }
    for (int old_h_index = 0; old_h_index < this->hidden_state_size; old_h_index++) {
      g_temp_cur += U[2 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + old_h_index]
          * h_queue[t_1][old_h_index];
    }
    g_temp_cur += b[2 * hidden_state_size + h_index];
    g_temp_cur = tanh(g_temp_cur);
    g_temp.push_back(g_temp_cur);
  }


//  Compute o
  for (int h_index = 0; h_index < this->hidden_state_size; h_index++) {
    o_temp_cur = 0;
    for (int inp_index = 0; inp_index < this->input_size; inp_index++) {
//      std::cout << "O weigjt = " << W[h_index][3 * this->input_size + inp_index] << std::endl;
      o_temp_cur += W[3 * this->input_size * hidden_state_size + h_index * input_size + inp_index] * inputs[inp_index];
    }
    for (int old_h_index = 0; old_h_index < this->hidden_state_size; old_h_index++) {
      o_temp_cur += U[3 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + old_h_index]
          * h_queue[t_1][old_h_index];
    }
    o_temp_cur += b[3 * hidden_state_size + h_index];
    o_temp_cur = sigmoid(o_temp_cur);
    o_temp.push_back(o_temp_cur);
  }
  f_queue[t] = f_temp;
  g_queue[t] = g_temp;
  i_queue[t] = i_temp;
  o_queue[t] = o_temp;
  x_queue[t] = inputs;

  float c_temp_float, h_temp_float;
  c_temp_float = h_temp_float = 0;
  for (int h_index = 0; h_index < this->hidden_state_size; h_index++) {
    c_queue[t][h_index] = f_queue[t][h_index] * c_queue[t_1][h_index] + i_queue[t][h_index] * g_queue[t][h_index];
    h_queue[t][h_index] = o_queue[t][h_index] * tanh(c_queue[t][h_index]);
    if (std::isnan(h_queue[t][h_index])) {
      for (int it = 0; it < truncation; it++) {
        std::cout << "T = " << it << ": ";
        print_vector(h_queue[t]);
      }
      std::cout << "NAN happening\n";
      exit(1);
    }
  }
//
  float prediction_temp = 0;
  for (int counter = 0; counter < prediction_weights.size(); counter++) {
    prediction_temp += h_queue[t][counter] * prediction_weights[counter];
  }
  this->time_step++;
  return prediction_temp;
}

std::vector<float> DenseLSTM::get_normalized_state() {
  int t = this->time_step % truncation;
  std::vector<float> my_vec;
  my_vec.reserve(prediction_weights.size());
  for (int i = 0; i < prediction_weights.size(); i++) {
    my_vec.push_back(h_queue[t][i]);
  }
  return my_vec;
}

void DenseLSTM::backward(int layer) {

//  For the last step, only gradient from the prediction flows backwards
  std::vector<std::vector<float>> initial_grad;
  initial_grad.push_back(this->prediction_weights);
  std::vector<float> c_inital(this->prediction_weights.size(), 0);
  initial_grad.push_back(c_inital);

//  Gradient for the linear predictor weightss
  int t = (this->time_step - 1) % truncation;
  auto h_cur = h_queue[t];
  for (int counter = 0; counter < h_cur.size(); counter++) {
    if (std::isnan(h_cur[counter])) {
      print_vector(h_cur);
      for (int it = 0; it < truncation; it++) {
        std::cout << "T = " << it << ": ";
        print_vector(h_queue[t]);
      }
      std::cout << "NAN happening\n";
      exit(1);
    }
    this->prediction_weights_grad[counter] += h_cur[counter];
  }

//  BPTT for k = this->truncation
  std::vector<std::vector<float>> grad_state = initial_grad;
  for (int counter = 0; counter < truncation; counter++) {
    t = (this->time_step - counter - 1) % truncation;
    if (t < 0)
      t += truncation;
    grad_state = this->backward_with_future_grad(grad_state, t);
  }
}

std::vector<std::vector<float>> DenseLSTM::backward_with_future_grad(std::vector<std::vector<float>> grad_prev_combined,
                                                                     int time) {

  auto grad_f = grad_prev_combined[0];
  auto grad_c = grad_prev_combined[1];
  std::vector<float> grad_h_new(hidden_state_size, 0);
  std::vector<float> grad_c_new(hidden_state_size, 0);
  int t_1 = time;
  int t_2 = t_1 - 1;
  if (t_2 < 0)
    t_2 = t_2 + truncation;
  std::vector<float> state_val;
  auto h = h_queue[t_1];
  auto c = c_queue[t_1];
  auto c_1 = c_queue[t_2];
  auto h_1 = h_queue[t_2];


//  Add gradient for i
  for (int h_index = 0; h_index < hidden_state_size; h_index++) {

    float grad = 0;
    float o_t = o_queue[t_1][h_index];
    float g_t = g_queue[t_1][h_index];
    float f_t = f_queue[t_1][h_index];
    float i_t = i_queue[t_1][h_index];

    float c_t = c[h_index];
    float c_t_1 = c_1[h_index];
    float h_t_1 = h_1[h_index];
    float tanh_c_t = tanh(c_t);
    float previous_grad = grad_f[h_index];
    float previous_c_grad = grad_c[h_index];
//    std::cout << "Prev grad = " << previous_grad << std::endl;

    float tanh_c_t_grad_o_t = (1 - tanh_c_t * tanh_c_t) * o_t;
    float g_t_i_t_1_i_t = g_t * i_t * (1 - i_t);
    float c_t_1_f_t_1_f_t = c_t_1 * f_t * (1 - f_t);

    float grad_i_base = (previous_grad * tanh_c_t_grad_o_t + previous_c_grad) * g_t_i_t_1_i_t;
    float grad_f_base = (previous_grad * tanh_c_t_grad_o_t + previous_c_grad) * c_t_1_f_t_1_f_t;
    float grad_g_base = (previous_grad * tanh_c_t_grad_o_t + previous_c_grad) * i_t * (1 - g_t * g_t);
    float grad_o_base = previous_grad * tanh(c_t) * o_t * (1 - o_t);

    for (int i = 0; i < input_size; i++) {
      float x_t = x_queue[t_1][i];

//      Grad for i_t
      W_grad[0 * this->input_size * hidden_state_size + h_index * input_size + i] += grad_i_base * x_t;
//      grad_h_new[h_index]
//      Grad for f_t
      W_grad[1 * this->input_size * hidden_state_size + h_index * input_size + i] += grad_f_base * x_t;

//      Grad for g_t
      W_grad[2 * this->input_size * hidden_state_size + h_index * input_size + i] += grad_g_base * x_t;

//    Grad for o_t
      W_grad[3 * this->input_size * hidden_state_size + h_index * input_size + i] += grad_o_base * x_t;
    }
    for (int i = 0; i < hidden_state_size; i++) {
      float hidden_t_1 = h_1[i];
//      Grad for Ui_t
      U_grad[0 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + i] +=
          grad_i_base * hidden_t_1;

//      Grad for Uf_t
      U_grad[1 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + i] +=
          grad_f_base * hidden_t_1;

//      Grad for Ug_t
      U_grad[2 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + i] +=
          grad_g_base * hidden_t_1;

//    Grad for Uo_t
      U_grad[3 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + i] +=
          grad_o_base * hidden_t_1;

      grad_h_new[i] +=
          grad_i_base * U[0 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + i];
      grad_h_new[i] +=
          grad_f_base * U[1 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + i];
      grad_h_new[i] +=
          grad_g_base * U[2 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + i];
      grad_h_new[i] +=
          grad_o_base * U[3 * this->hidden_state_size * hidden_state_size + h_index * hidden_state_size + i];
    }

    grad_c_new[h_index] += (previous_grad * tanh_c_t_grad_o_t + previous_c_grad) * f_t;

//      Grad for bi_t
    b_grad[0 * hidden_state_size + h_index] += grad_i_base;

//      Grad for Bf_t
    b_grad[1 * hidden_state_size + h_index] += grad_f_base;

//      Grad for Bg_t
    b_grad[2 * hidden_state_size + h_index] += grad_g_base;

//    Grad for Bo_t
    b_grad[3 * hidden_state_size + h_index] += grad_o_base;

  }
  std::vector<std::vector<float>> return_vec;
  return_vec.push_back(grad_h_new);
  return_vec.push_back(grad_c_new);
//  std::cout << "grad w.r.t to H is\n";
//  print_vector(grad_h_new);
  return return_vec;
}
std::vector<float> DenseLSTM::get_state() {
  int t_1 = (this->time_step - 1) % truncation;
  std::vector<float> state_val;
  if (t_1 < 0)
    t_1 = truncation + t_1;
  for (int i = 0; i < h_queue[t_1].size(); i++)
    state_val.push_back(h_queue[t_1][i]);
  for (int i = 0; i < c_queue[t_1].size(); i++)
    state_val.push_back(c_queue[t_1][i]);
  return state_val;
}

void DenseLSTM::decay_gradient(float decay_rate) {
  for (int counter = 0; counter < this->prediction_weights_grad.size(); counter++)
    this->prediction_weights_grad[counter] *= decay_rate;
  for (int counter = 0; counter < W_grad.size(); counter++)
    W_grad[counter] *= decay_rate;
  for (int counter = 0; counter < U_grad.size(); counter++)
    U_grad[counter] *= decay_rate;
  for (int counter = 0; counter < b_grad.size(); counter++)
    b_grad[counter] *= decay_rate;
}

void DenseLSTM::zero_grad() {
  for (int counter = 0; counter < this->prediction_weights_grad.size(); counter++)
    this->prediction_weights_grad[counter] = 0;
  for (int counter = 0; counter < W_grad.size(); counter++)
    W_grad[counter] = 0;
  for (int counter = 0; counter < U_grad.size(); counter++)
    U_grad[counter] = 0;
  for (int counter = 0; counter < b_grad.size(); counter++)
    b_grad[counter] = 0;
}

void DenseLSTM::update_parameters(int layer, float error) {
  for (int counter = 0; counter < this->prediction_weights_grad.size(); counter++)
    this->prediction_weights[counter] += this->prediction_weights_grad[counter] * error * step_size;
  for (int counter = 0; counter < W_grad.size(); counter++)
    W[counter] += step_size * error * W_grad[counter];
  for (int counter = 0; counter < U_grad.size(); counter++)
    U[counter] += step_size * error * U_grad[counter];
  for (int counter = 0; counter < b_grad.size(); counter++)
    b[counter] += step_size * error * b_grad[counter];
}
