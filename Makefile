CXX      = mpicxx
CXXFLAGS = -std=c++17 -fopenmp $(shell pkg-config --cflags opencv4)
LDFLAGS  = -fopenmp $(shell pkg-config --libs opencv4)

TARGET = main
SRC    = main.cpp

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run: all
	mpirun -np 4 ./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf frames/