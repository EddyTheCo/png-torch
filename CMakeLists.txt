cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(png-torch VERSION 0.1 DESCRIPTION "interface torch tensors to images" LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(PNG REQUIRED)

add_library(png-torch png-torch.cpp include/utils/png-torch.hpp)

target_link_libraries(png-torch PUBLIC  "${TORCH_LIBRARIES}")
target_link_libraries(png-torch PRIVATE  PNG::PNG)


target_include_directories(png-torch PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

install(TARGETS png-torch EXPORT png-torchTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT png-torchTargets
	FILE png-torchTargets.cmake
	DESTINATION lib/cmake/png-torch
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/png-torchConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/png-torch"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/png-torchConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/png-torchConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/png-torchConfigVersion.cmake
	DESTINATION lib/cmake/png-torch
	)
export(EXPORT png-torchTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/png-torchTargets.cmake"
	)
