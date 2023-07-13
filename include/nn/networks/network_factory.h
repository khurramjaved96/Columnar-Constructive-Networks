//
// Created by Khurram Javed on 2022-07-28.
//

#ifndef INCLUDE_NN_NETWORKS_NETWORK_FACTORY_H_
#define INCLUDE_NN_NETWORKS_NETWORK_FACTORY_H_

#include "base_lstm.h"
#include <string>
#include "../../experiment/Experiment.h"

class NetworkFactory{
 public:
  static BaseLSTM* get_network(Experiment* exp_config);
};

#endif //INCLUDE_NN_NETWORKS_NETWORK_FACTORY_H_
