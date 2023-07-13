add_subdirectory(supported)
target_sources(AtariExperiment
  PRIVATE
    Roms.cpp
    RomSettings.cpp
    RomUtils.cpp
)
