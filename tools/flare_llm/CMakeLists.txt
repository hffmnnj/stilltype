cmake_minimum_required(VERSION 3.16)
project(stilltype_flare_llm LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The memory preflight requires Linux /proc/meminfo")
endif()
if(NOT IS_DIRECTORY "${LLAMA_CPP_SOURCE}" OR NOT EXISTS "${LLAMA_CPP_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "Set LLAMA_CPP_SOURCE to the trusted llama.cpp source at revision d81235049384534c167caea52b85a694f6103d14")
endif()

foreach(option_name BUILD_SHARED_LIBS LLAMA_BUILD_COMMON LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS LLAMA_BUILD_EXAMPLES
        LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_OPENSSL GGML_CUDA GGML_VULKAN GGML_SYCL
        GGML_METAL GGML_OPENCL GGML_HIP GGML_MUSA GGML_RPC GGML_WEBGPU GGML_BACKEND_DL)
    set(${option_name} OFF CACHE BOOL "Disabled for the CPU research utility" FORCE)
endforeach()
add_subdirectory("${LLAMA_CPP_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/llama" EXCLUDE_FROM_ALL)

add_executable(stilltype-flare-llm main.cpp)
target_compile_features(stilltype-flare-llm PRIVATE cxx_std_17)
target_link_libraries(stilltype-flare-llm PRIVATE llama)

add_executable(flare-llm-selection-self-test self_test.cpp)
target_compile_features(flare-llm-selection-self-test PRIVATE cxx_std_17)
# Upstream defaults to Release, but these behavioral assertions must remain active.
target_compile_options(flare-llm-selection-self-test PRIVATE -UNDEBUG)
enable_testing()
add_test(NAME flare-llm-selection COMMAND flare-llm-selection-self-test)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME flare-llm-output-privacy
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_output.py" "$<TARGET_FILE:stilltype-flare-llm>")
