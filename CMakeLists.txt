cmake_minimum_required(VERSION 3.11)
project(swift_f0_cpp)

# static
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

# realtime
add_executable(realtime realtime.cpp miniaudio.c)
set_target_properties(realtime PROPERTIES CXX_STANDARD 20)
set_target_properties(realtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

target_include_directories(realtime PUBLIC onnx/include)
target_link_directories(realtime PUBLIC onnx/lib)
target_link_libraries(realtime PUBLIC onnxruntime onnxruntime_providers_shared)

target_include_directories(realtime PUBLIC raylib/include)
target_link_directories(realtime PUBLIC raylib/lib)
target_link_libraries(realtime PUBLIC raylib winmm.lib)
