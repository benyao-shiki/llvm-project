CXX=g++
CXX_FLAGS=-I/usr/local/cuda/include -std=c++11 -fPIC -shared
LD_FLAGS=-ldl -lcuda

BUILD_DIR=build
TARGET=$(BUILD_DIR)/libcheckkernel.so
SRC=checkkernel.cpp
OBJ=$(BUILD_DIR)/$(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -o $@ $(OBJ) $(LD_FLAGS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
