//
// Created by Khurram Javed on 2022-07-19.
//

#include "../../include/environments/proto_prediction_environments.h"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
ProtoPredictionEnvironment::ProtoPredictionEnvironment(std::string game_name,
                                                       float gamma)
    : gray_features(210 * 160, 0), observation(275, 0) {

  this->gamma = gamma;
  this->gamma = gamma;
  std::fstream policy("../policies/" + game_name + "NoFrameskip-v4.txt",
                      std::ios::in | std::ios::binary);
  my_env.setInt("random_seed", 1731038949);
  //  my_env.setBool("truncate_on_loss_of_life", true);
  my_env.setFloat("repeat_action_probability", 0.0);
  my_env.setInt("frame_skip", 1);
  my_env.loadROM("../games/" + game_name + ".bin");
  my_env.reset_game();

  long size;
  policy.seekg(0, std::ios::end);
  size = policy.tellg();
  policy.seekg(0, std::ios::beg);
  actions = new char[size];
  policy.read(actions, size);
  policy.close();
  std::cout << "Size of actions = " << size << std::endl;
  action_set = my_env.getMinimalActionSet();
  time = 0;
  reward = 0;
  ep_reward = 0;
  to_reset = false;
}

AtariLarge::AtariLarge(std::string path, float gamma)
    : ProtoPredictionEnvironment(path, gamma) {
  this->observation = std::vector<float>(50 * 50 + 19, 0);
}

std::vector<float> ProtoPredictionEnvironment::get_state() {
  return this->observation;
}
void ProtoPredictionEnvironment::UpdateReturns() {
  float old_val = 0;
  list_of_returns = std::vector<float>(list_of_rewards.size(), 0);
  for (int i = list_of_rewards.size() - 1; i >= 0; i--) {
    list_of_returns[i] = list_of_rewards[i] + old_val;
    old_val = list_of_returns[i] * this->gamma;
  }
  list_of_rewards.clear();
}
// S, 1, S, 1, S, R,
std::vector<float> &ProtoPredictionEnvironment::GetListOfReturns() {
  return this->list_of_returns;
}
bool ProtoPredictionEnvironment::get_done() { return true; }
std::vector<float> ProtoPredictionEnvironment::step() {
  to_reset = false;
  time++;
  for (int i = 256; i < 275; i++) {
    observation[i] = 0;
  }
  if (actions[time] == 'R') {
    //    std::cout << "Episode reward = " << ep_reward << std::endl;
    reward = 0;
    list_of_rewards.push_back(0);
    my_env.reset_game();
    ep_reward = 0;
    UpdateReturns();
    to_reset = true;
  } else {
    reward = my_env.act(action_set[int(actions[time]) - 97]);
    ep_reward += reward;
    list_of_rewards.push_back(get_reward());
    observation[int(actions[time]) - 97 + 256] = 1;
    observation[274] = get_reward();
  }
  my_env.getScreenGrayscale(gray_features);
  cv::Mat image(210, 160, CV_8UC1, gray_features.data());
  cv::Mat dest;
  cv::resize(image, dest, cv::Size(16, 16));
  for (int i = 0; i < 16 * 16; i++) {
    observation[i] = float(dest.data[i]) / 256;
  }
  return this->get_state();
}

//std::vector<float> AtariLarge::step() {
//  return observation;
//}

std::vector<float> AtariLarge::step() {
  to_reset = false;
  time++;
  for (int i = 50 * 50; i < 50 * 50 + 19; i++) {
    observation[i] = 0;
  }
  if (actions[time] == 'R') {
    //    std::cout << "Episode reward = " << ep_reward << std::endl;
    reward = 0;
    list_of_rewards.push_back(0);
    my_env.reset_game();
    ep_reward = 0;
    UpdateReturns();
    to_reset = true;
  } else {
    reward = my_env.act(action_set[int(actions[time]) - 97]);
    ep_reward += reward;
    list_of_rewards.push_back(get_reward());
    observation[int(actions[time]) - 97 + 50 * 50] = 1;
    observation[50 * 50 + 18] = get_reward();
  }
  my_env.getScreenGrayscale(gray_features);
  cv::Mat image(210, 160, CV_8UC1, gray_features.data());
  cv::Mat dest;
  cv::resize(image, dest, cv::Size(50, 50));
  for (int i = 0; i < 50 * 50; i++) {
    observation[i] = float(dest.data[i]) / 256;
  }
  return this->get_state();
}

float ProtoPredictionEnvironment::get_target() { return real_target[time]; }

float ProtoPredictionEnvironment::get_gamma() {
  if (to_reset)
    return 0;
  return gamma;
}
float ProtoPredictionEnvironment::get_reward() {
  if (reward > 1)
    return 1;
  else if (reward < -1)
    return -1;
  return reward;
}
