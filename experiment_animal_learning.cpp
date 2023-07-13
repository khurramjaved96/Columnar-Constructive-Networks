//
// Created by Khurram Javed on 2022-01-08.
//

#include <iostream>
#include "include/nn/networks/lstm_incremental_networks.h"
#include "include/nn/networks/lstm_bptt.h"
#include "include/utils.h"
#include "include/nn/utils.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"
#include <string>
#include <deque>
#include "include/nn/networks/network_factory.h"
#include "include/nn/networks/base_lstm.h"
#include "include/environments/animal_learning/tracecondioning.h"

int main(int argc, char *argv[]) {

  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  Metric error_metric = Metric(my_experiment->database_name, "error_table",
                               std::vector<std::string>{"run", "step", "error"},
                               std::vector<std::string>{"int", "int", "real"},
                               std::vector<std::string>{"run", "step"});

  Metric avg_error = Metric(my_experiment->database_name, "predictions",
                            std::vector<std::string>{"run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "pred",
                                                     "target"},
                            std::vector<std::string>{"int", "int", "real", "real", "real", "real", "real", "real",
                                                     "real", "real", "real"},
                            std::vector<std::string>{"run", "step"});
//  Metric network->state = Metric(my_experiment->database_name, "network->state",
//                                std::vector<std::string>{"run", "step", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
//                                                         "x8", "x9"},
//                                std::vector<std::string>{"int", "int", "real", "real", "real", "real", "real", "real",
//                                                         "real", "real", "real", "real"},
//                                std::vector<std::string>{"run", "step"});

  std::cout << "Program started \n";

  std::mt19937 mt(my_experiment->get_int_param("seed"));
  std::pair<int, int> ISI{24, 36};
  std::pair<int, int> ITI{80, 120};
  TracePatterning env = TracePatterning(ISI, ITI, 5, my_experiment->get_int_param("seed"));

  BaseLSTM *network = NetworkFactory::get_network(my_experiment);
  int K = 100000;
  std::cout << "Network created\n";
  long double running_error = 0;
  std::deque<float> list_of_errors;
  float gamma = 0.90;
  auto x = env.reset();
  float old_pred = network->forward(x);
  int layer = 0;
  for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {

    if (i % std::stoi(my_experiment->get_vector_param("features")[2]) == std::stoi(my_experiment->get_vector_param("features")[2]) - 1) {
      layer++;
      std::cout << "Increasing layer\n";
    }

    x = env.step();

    network->decay_gradient(my_experiment->get_float_param("lambda") *
                            gamma);
    network->backward(layer);

    float pred = network->forward(x);
    float real_target = env.get_target(gamma);
    float real_error = (real_target - pred) * (real_target - pred);
    float target =  env.get_US()  + gamma * pred;
    float error = target - old_pred;

    network->update_parameters(layer, error);
    old_pred = pred;

    if (i % 1000000 < 400) {
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      for (int inner_c = 0; inner_c < 7; inner_c++) {
        cur_error.push_back(std::to_string(x[inner_c]));
      }
      cur_error.push_back(std::to_string(pred));
      cur_error.push_back(std::to_string(real_target));
      avg_error.record_value(cur_error);
    }


    list_of_errors.push_back(real_error);
    running_error += real_error;
    if(list_of_errors.size() > K){
      running_error -= list_of_errors[0];
      list_of_errors.pop_front();
    }
    if(list_of_errors.size() > K){
      std::cout << "Queue longer than 100000; shouldn't happen. Exiting\n";
      exit(0);
    }

    if (i % 50000 == 20000 and i > K) {
      std::vector<std::string> cur_error;
      cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
      cur_error.push_back(std::to_string(i));
      cur_error.push_back(std::to_string(running_error/K));
      error_metric.record_value(cur_error);
    }
//
    if (i % 500000 == 0) {
//      IncrementalNetworks* ptr = dynamic_cast<IncrementalNetworks*>(network);
//      if(ptr!= nullptr) {
//        print_vector(ptr->prediction_weights);
//      }
      std::cout << "Step = " << i << " Error = " << running_error/K << std::endl;
      error_metric.commit_values();
      avg_error.commit_values();
    }
  }
  error_metric.commit_values();
  avg_error.commit_values();

}
