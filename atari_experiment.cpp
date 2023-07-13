//
// Created by Khurram Javed on 2022-07-18.
//

#include "include/environments/proto_prediction_environments.h"
#include "include/utils.h"
#include <fstream>
#include <iostream>
#include <vector>

#include "include/environments/animal_learning/tracecondioning.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include "include/nn/networks/base_lstm.h"
#include "include/nn/networks/lstm_bptt.h"
#include "include/nn/networks/lstm_incremental_networks.h"
#include "include/nn/networks/network_factory.h"
#include "include/nn/utils.h"
#include "include/utils.h"
#include <deque>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  std::vector<float> list_of_returns;
  std::vector<float> list_of_predictions;
  std::deque<float> list_of_errors;
  int K = 500000;
  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  ProtoPredictionEnvironment env2(my_experiment->get_string_param("env"),
                                  my_experiment->get_float_param("gamma"));

  Metric error_metric = Metric(my_experiment->database_name, "error_table",
                               std::vector<std::string>{"run", "step", "error"},
                               std::vector<std::string>{"int", "int", "real"},
                               std::vector<std::string>{"run", "step"});

  Metric episodic_error =
      Metric(my_experiment->database_name, "episodic_error",
             std::vector<std::string>{"run", "step", "error"},
             std::vector<std::string>{"int", "int", "real"},
             std::vector<std::string>{"run", "step"});

  Metric avg_error = Metric(
      my_experiment->database_name, "predictions",
      std::vector<std::string>{"run", "step", "pred", "target", "reward"},
      std::vector<std::string>{"int", "int", "real", "real", "real"},
      std::vector<std::string>{"run", "step"});

  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment->get_int_param("seed"));

  BaseLSTM *network = NetworkFactory::get_network(my_experiment);
  std::cout << "Network created\n";
  float running_error = 0.00;

  auto x = env2.step();
  int layer = 0;
  float old_pred = network->forward(x);
  list_of_predictions.push_back(old_pred);
  for (int env_step = 1; env_step < my_experiment->get_int_param("steps");
       env_step++) {
    if (env_step % std::stoi(my_experiment->get_vector_param("features")[2]) == std::stoi(my_experiment->get_vector_param("features")[2]) - 1) {
      layer++;
      std::cout << "Increasing layer\n";
    }
    if(env_step%100000==0) {
      std::cout << "Step = " << env_step << "\n";
      IncrementalNetworks* ptr = dynamic_cast<IncrementalNetworks*>(network);
      if(ptr!= nullptr)
      print_vector(ptr->prediction_weights);
    }

    x = env2.step();
//    Evaluation code
    if (env2.get_gamma() == 0) {
      auto returns = env2.GetListOfReturns();
      float error = 0;
      for (int i = 1; i < returns.size(); i++) {
        int actual_env_step = env_step - returns.size() + i;
        float real_error = (returns[i] - list_of_predictions[i]) *
                           (returns[i] - list_of_predictions[i]);
        list_of_errors.push_back(real_error);
        running_error += real_error;
        if (list_of_errors.size() > K) {
          running_error -= list_of_errors[0];
          list_of_errors.pop_front();
        }
        error += real_error;
        if (actual_env_step % 50000 == 0 && list_of_errors.size() == K) {
          std::vector<std::string> cur_error;
          cur_error.push_back(
              std::to_string(my_experiment->get_int_param("run")));
          cur_error.push_back(std::to_string(actual_env_step));
          cur_error.push_back(std::to_string(running_error/K));
          std::cout << "Running error = " << running_error/K << std::endl;
          error_metric.record_value(cur_error);
        }
        if (actual_env_step % 1000000 < 500) {
          std::vector<std::string> cur_error;
          cur_error.push_back(
              std::to_string(my_experiment->get_int_param("run")));
          cur_error.push_back(std::to_string(env_step + i));
          cur_error.push_back(std::to_string(list_of_predictions[i]));
          cur_error.push_back(std::to_string(returns[i]));
          cur_error.push_back(std::to_string(0));
          avg_error.record_value(cur_error);
        }
      }
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
      cur_error.push_back(std::to_string(env_step));
      cur_error.push_back(std::to_string(error / returns.size()));
      episodic_error.record_value(cur_error);

      std::cout << "Episode return error = " << error / returns.size()
                << std::endl;
      list_of_predictions.clear();
    }
//    Evaluation code ends

    network->decay_gradient(my_experiment->get_float_param("lambda") *
                            env2.get_gamma());
    network->backward(layer);

    float pred = network->forward(x);
    list_of_predictions.push_back(pred);

    float target = env2.get_reward() + env2.get_gamma() * pred;
    float error = target - old_pred;
    network->update_parameters(layer, error);
    if (env_step % 100000 == 0) {
      error_metric.commit_values();
      avg_error.commit_values();
      episodic_error.commit_values();
    }
    old_pred = pred;
  }
  error_metric.commit_values();
  avg_error.commit_values();
  episodic_error.commit_values();
}