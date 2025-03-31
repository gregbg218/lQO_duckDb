cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME logqo)

# Find the required packages
find_package(Torch REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)

set(EXTENSION_SOURCES src/logqo_extension.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link Torch in both the static library and the loadable extension
target_link_libraries(${EXTENSION_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${LOADABLE_EXTENSION_NAME} ${TORCH_LIBRARIES})

# Make sure we use C++14 which is needed for PyTorch
set_property(TARGET ${EXTENSION_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${LOADABLE_EXTENSION_NAME} PROPERTY CXX_STANDARD 14)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
