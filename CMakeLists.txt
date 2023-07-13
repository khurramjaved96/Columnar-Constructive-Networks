cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

add_executable(AtariExperiment atari_experiment.cpp  src/environments/proto_prediction_environments.cpp  src/utils.cpp
      src/nn/networks/lstm_bptt.cpp include/json.hpp  src/experiment/Experiment.cpp
        src/experiment/Database.cpp src/experiment/Metric.cpp
         src/nn/synapse.cpp
        src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp
        src/nn/networks/neural_network.cpp src/nn/utils.cpp src/utils.cpp
         src/environments/animal_learning/traceconditioning.cpp
        src/environments/animal_learning/tracepatterning.cpp  src/nn/networks/lstm_incremental_networks.cpp
        src/nn/lstm.cpp  src/nn/lstm.cpp  src/nn/networks/network_factory.cpp)

add_executable(AnimalLearning src/nn/networks/network_factory.cpp src/nn/networks/lstm_bptt.cpp
        include/json.hpp  src/experiment/Experiment.cpp
        src/experiment/Database.cpp src/experiment/Metric.cpp
         src/nn/synapse.cpp
        src/nn/neuron.cpp src/nn/utils.cpp src/nn/dynamic_elem.cpp
        src/nn/networks/neural_network.cpp src/nn/utils.cpp src/utils.cpp
         src/environments/animal_learning/traceconditioning.cpp
        src/environments/animal_learning/tracepatterning.cpp src/nn/networks/lstm_incremental_networks.cpp src/nn/lstm.cpp   src/nn/lstm.cpp experiment_animal_learning.cpp)


add_executable(Count include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
        count_configs.cpp)


target_link_libraries(AtariExperiment PUBLIC tbb)
target_link_libraries(AtariExperiment PUBLIC mariadb  ${OpenCV_LIBS}  ZLIB::ZLIB
        Threads::Threads dl)

target_link_libraries(AnimalLearning PUBLIC tbb)
target_link_libraries(AnimalLearning PUBLIC mariadb)

target_link_libraries(Count PUBLIC tbb)
target_link_libraries(Count PUBLIC mariadb)


project(FlexibleNN LANGUAGES CXX)

