cmake_minimum_required(VERSION 3.2.0)

file(GLOB sources ./*.cpp)

add_library(find_halcon_calbrd STATIC ${sources})

find_package(OpenCV REQUIRED core imgproc calib3d)
if (OpenCV_FOUND)
    target_include_directories(find_halcon_calbrd PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(find_halcon_calbrd PRIVATE ${OpenCV_LIBS})
endif()

target_include_directories(find_halcon_calbrd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

