cmake_minimum_required(VERSION 3.2.0)

project(find_calib_pattern)

file(GLOB cpp_sources ./src/*.cpp)

add_library(find_calib_pattern STATIC ${cpp_sources})

find_package(OpenCV REQUIRED core calib3d)
if (OpenCV_FOUND)
    target_include_directories(find_calib_pattern PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(find_calib_pattern PRIVATE ${OpenCV_LIBS})
endif()

target_include_directories(find_calib_pattern PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ----------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
