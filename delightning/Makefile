CXX=g++
CXXFLAGS=-std=c++17 -Wall -Wextra -O2

TARGET=delightning
SRCS=src/main.cpp

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
