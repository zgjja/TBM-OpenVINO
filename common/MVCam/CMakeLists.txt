cmake_minimum_required(VERSION 3.10)
project(cam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
aux_source_directory("./src" CAM_SRC)

add_library(${PROJECT_NAME} SHARED ${CAM_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libMVSDK.so
    )
