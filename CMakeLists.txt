cmake_minimum_required(VERSION 3.10)
project(Test)
find_package(OpenCV REQUIRED)

add_executable(Basic basicAssignment.cpp)


target_include_directories(
    Basic
    PUBLIC ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Basic ${OpenCV_LIBS})