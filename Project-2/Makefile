#! /usr/bin/make

OPT ?= -O2
STD ?= -std=c++20

SOURCES = $(wildcard *.cpp)
TARGETS = $(SOURCES:.cpp=.out)

CXXFLAGS = $(OPT) $(STD)

# Select files that should be removed when we need to "clean" a project
DIRT = $(wildcard *.o *.out *.dSYM)

#----------------------------------------------------------------------------

default: targets

targets : $(TARGETS)

%.out : %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

#----------------------------------------------------------------------------

clean:
	$(RM) $(DIRT)

.PHONY: default targets clean 