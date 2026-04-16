CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall

TARGET   = runtest
SRCS     = runtest.cpp planner_multirobot.cpp
OBJS     = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET) robot_trajectory_*.txt simulation.gif map_preview.png

.PHONY: all clean