//
// Created by Khurram Javed on 2023-07-11.
//

#include "../../include/environments/weather.h"
#include "../../include/utils.h"
#include "../../include/rapidcsv.h"
#include <fstream>
#include <string>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>

WeatherPrediction::WeatherPrediction(std::string game_name) {
  file_name = game_name;
  std::ifstream myfile(file_name);
  f_pointer = myfile.tellg();
  myfile.close();
}

void WeatherPrediction::read_lines(int n) {
  std::ifstream myfile;
  while(true) {
//    std::cout << file_name << std::endl;
    myfile = std::ifstream(file_name);
    if(myfile.is_open())
      break;
    else{
      std::cout << "Sleeping\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
  }
  myfile.seekg(f_pointer);
  std::string line;
  if(myfile.is_open()) {
    int i = 0;
    while(getline(myfile, line)){
      std::vector<float> data;
      std::string cur_letter = "";
      bool abanton_line = false;
      for (int j = 0; j < line.size(); j++) {
        if (line[j] == ',') {
          if (cur_letter.size() > 0) {
            float temp = std::stof(cur_letter);
            data.push_back(temp);
            cur_letter = "";
          } else {
            abanton_line = true;
          }
        } else {
          cur_letter += line[j];
        }
      }
      if (!abanton_line)
        data_buffer.push(data);
      i++;
      if(i%n==0){
        break;
      }
    }
    f_pointer = myfile.tellg();
    myfile.close();
  }
}

std::vector<float> WeatherPrediction::step() {
  if(data_buffer.size() == 0){
    read_lines(1000000);
  }
  auto x = data_buffer.front();
  data_buffer.pop();
  return x;
}