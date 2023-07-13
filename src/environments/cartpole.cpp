//
// Created by Khurram Javed on 2021-10-06.
//

#include "../../include/environments/cartpole.h"
#include <math.h>

CartPole::CartPole() {
  this->gravity = 9.8;
  this->masscart = 1.0;
  this->masspole = 0.1;
  this->total_mass = this->masspole * this->masscart;
  this->length = 0.5;
  this->polemass_length = this->masspole * this->masscart;
  this->force_mag = 10.0;
  this->tau = 0.02;
  this->theta_threshold_radians = 12*2*(M_PI/360.0);
  this->x_threshold = 2.4;
  this->total_actions = 2;
}


int CartPole::get_no_of_actions() {
  return total_actions;
}