
.PHONY: all configure build clean run
MAKEFLAGS += --silent

all: run

configure:
	mkdir -p build
	cmake -B build

build: configure
	cmake --build build

run: build
	./build/neural

clean:
	rm -rf build
