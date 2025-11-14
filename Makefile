// # Compiler and flags
// NVCC = nvcc
// CUDA_ARCH = -arch=sm_70
// NCCL_PATH = /usr/local/nccl
// 
// CXXFLAGS = -O3 -std=c++11
// INCLUDES = -I./include -I$(NCCL_PATH)/include
// LDFLAGS = -L$(NCCL_PATH)/lib -lnccl
// 
// # Directories
// SRC_DIR = src
// BUILD_DIR = build
// BIN_DIR = bin
// 
// # Source files
// SOURCES = $(SRC_DIR)/main.cu \
//           $(SRC_DIR)/game_of_life.cu \
//           $(SRC_DIR)/gpu_manager.cu \
//           $(SRC_DIR)/kernels.cu
// 
// # Target executable
// TARGET = $(BIN_DIR)/game_of_life
// 
// all: directories $(TARGET)
// 
// directories:
// 	@mkdir -p $(BUILD_DIR) $(BIN_DIR)
// 
// $(TARGET): $(SOURCES)
// 	$(NVCC) $(CUDA_ARCH) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)
// 
// clean:
// 	rm -rf $(BUILD_DIR) $(BIN_DIR)
// 
// run: $(TARGET)
// 	./$(TARGET)
// 
// .PHONY: all clean run directories