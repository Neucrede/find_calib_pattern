file(GLOB sources
    ./*.cpp
)

add_executable(test ${sources})
target_link_libraries(test PRIVATE find_calib_pattern)

find_package(OpenCV REQUIRED core videoio imgproc)
if (OpenCV_FOUND)
    target_include_directories(test PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test PRIVATE ${OpenCV_LIBS})
endif()

include(GNUInstallDirs)

install(
    TARGETS test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

