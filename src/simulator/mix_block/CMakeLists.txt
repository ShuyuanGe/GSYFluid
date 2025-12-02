file(GLOB simulator_mix_block_src ./*.cpp ./*.cu)

add_library(simulator_mix_block STATIC ${simulator_mix_block_src})

target_include_directories(simulator_mix_block PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(simulator_mix_block PUBLIC lbm_core blocking_core)