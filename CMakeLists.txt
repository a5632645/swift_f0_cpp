cmake_minimum_required(VERSION 3.11)
project(swift_f0_cpp)

add_executable(swift_f0_cpp main.cpp)
set_target_properties(swift_f0_cpp PROPERTIES CXX_STANDARD 20)
set_target_properties(swift_f0_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

target_include_directories(swift_f0_cpp PUBLIC onnx/include)
target_link_directories(swift_f0_cpp PUBLIC onnx/lib)
target_link_libraries(swift_f0_cpp PUBLIC onnxruntime onnxruntime_providers_shared)

target_include_directories(swift_f0_cpp PUBLIC raylib/include)
target_link_directories(swift_f0_cpp PUBLIC raylib/lib)
target_link_libraries(swift_f0_cpp PUBLIC raylib winmm.lib)
