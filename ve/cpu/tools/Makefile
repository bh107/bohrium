BH_SRC=$(realpath ../../..)
CPU=$(BH_SRC)/ve/cpu
BDIR=/tmp/bhbuild_cpu/
BH_PYTHON=python
TEST_EXCLUDE=
DUMPSRC=1

UNAME=$(shell uname)
PAR_LVL=-j1
ifeq ($(UNAME), Darwin)
	PAR_LVL=-j$(shell sysctl -n hw.ncpu)
endif
ifeq ($(UNAME), Linux)
	PAR_LVL=-j$(shell expr $(shell nproc) / 2)
endif

BUILD_OPTIONS=

gnu:
	$(eval CC := gcc)
	$(eval CXX := g++)

pgi:
	$(eval CC := pgcc)
	$(eval CXX := g++)

path:
	$(eval CC := pathcc)
	$(eval CXX := g++)

intel:
	$(eval CC := icc)
	$(eval CXX := g++)

clang:
	$(eval CC := clang)
	$(eval CXX := clang++)

use_dython:
	$(eval BH_PYTHON := dython)

for_dython:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DPYTHON_EXECUTABLE=/usr/bin/dython -DPY_SCRIPT=python)

release_on:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DCMAKE_BUILD_TYPE=Release)

cil_off:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DBENCHMARK_CIL=OFF -DBRIDGE_CIL=OFF -DTEST_CIL=OFF -DBRIDGE_NUMCIL=OFF)

cluster_off:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DVEM_CLUSTER=OFF -DVEM_PROXY=OFF)

fun_off:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DEXT_VISUALIZER=OFF)

gpu_off:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DVE_GPU=OFF)

cpu_debug_on:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DVE_CPU_DEBUGGING=ON -DCMAKE_BUILD_TYPE=Debug)

cpu_profiling_on:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DVE_CPU_PROFILING=ON)

fuser_gentle_off:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DFUSER_GENTLE=OFF)

fuser_greedy_off:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DFUSER_GREEDY=OFF)

fuser_optimal_off:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DFUSER_OPTIMAL=OFF)

fuser_singleton_off:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DFUSER_SINGLETON=OFF)

fuser_topological_only: fuser_greedy_off fuser_optimal_off fuser_singleton_off

custom_boost_154:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DBOOST_ROOT=$(HOME)/tools/boost-1.54.0 -DBoost_INCLUDE_DIRS=$(HOME)/tools/boost-1.54.0/include -DBoost_LIBRARY_DIRS=$(HOME)/tools/boost-1.54.0/lib -DBoost_NO_SYSTEM_PATHS=ON -DBoost_NO_BOOST_CMAKE=ON )

custom_boost_158:
	$(eval BUILD_OPTIONS := $(BUILD_OPTIONS) -DBOOST_ROOT=$(HOME)/tools/boost-1.58.0 -DBoost_INCLUDE_DIRS=$(HOME)/tools/boost-1.58.0/include -DBoost_LIBRARY_DIRS=$(HOME)/tools/boost-1.58.0/lib -DBoost_NO_SYSTEM_PATHS=ON -DBoost_NO_BOOST_CMAKE=ON )


.PHONY: help

BLUE=\033[0;35m
NC=\033[0m

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[0;35m%-20s\033[0m %s\n", $$1, $$2}'


purge_ko: ## Purge runtime generated objects
	@echo "${BLUE}Purging runtime generated objects${NC}"
	rm -f ~/.local/var/bohrium/kernels/*
	rm -rf ~/.local/var/bohrium/objects/*
	rm -rf ~/.local/var/bohrium/fuse_cache/*


purge_cpu: purge_ko ## Remove installed stuff specific to the CPU-engine
	@echo "${BLUE}Removing installed stuff specific to the CPU-engine${NC}"
	rm -rf ~/.local/share/bohrium/templates/*
	rm -rf ~/.local/var/bohrium/scripts
	rm -rf ~/.local/lib/libbh_ve_cpu.*


purge_all: purge_cpu ## Remove everything (this also does 'git clean' and removes ~/.local)
	@echo "${BLUE}Removing everything${NC}"
	@echo "${BLUE}Running git clean${NC}"
	cd $(BH_SRC) && git clean -fdx &&	git clean -fdX
	@echo "${BLUE}Removing ~/.local${NC}"
	rm -rf $(BDIR)
	rm -rf ~/.local/bin
	rm -rf ~/.local/include
	rm -rf ~/.local/lib
	rm -rf ~/.local/share/bohrium
	rm -rf ~/.local/var


reset: purge_all clean ## Purge and clean


install: purge_ko ## Install the CPU-engine
	@echo "${BLUE}Installing the CPU-engine${NC}"
	mkdir -p $(BDIR)
	cd $(BDIR) &&	CC=$(CC) CXX=$(CXX) cmake $(BH_SRC) $(BUILD_OPTIONS) && make $(PAR_LVL) && make install


install_dev: cil_off cluster_off fun_off gpu_off install ## Install without CIL, Cluster, Fun, and GPU


install_debug: cil_off cluster_off fun_off gpu_off cpu_debug_on install ## Install without CIL, Cluster, Fun, and GPU but with debug


install_profiling: cil_off cluster_off fun_off gpu_off cpu_profiling_on install ## Install without CIL, Cluster, Fun, and GPU but with profiling


clean: ## Clean up the build-files
	@echo "${BLUE}Cleaning up the build-files${NC}"
	rm -rf $(BDIR) core trace-*.txt flow*.dot flow*.html graph-*.dot $(CPU)/doc/output


crap: ## Remove dot-, trace-, flow-, and core-files
	@echo "${BLUE}Removing dot-, trace-, flow-, and core-files${NC}"
	rm -f graph*.dot flow*.dot trace*.txt flow*.html vgcore.* core /tmp/graph*.html /tmp/graph*.png /tmp/graph*.png


test: ## Run tests with JIT 3 and 1
	@echo "${BLUE}Running tests with JIT 3${NC}"
	BH_CPU_JIT_LEVEL=3 $(BH_PYTHON) $(BH_SRC)/test/python/numpytest.py $(TEST_EXCLUDE)
	@echo "${BLUE}Running tests with JIT 1${NC}"
	BH_CPU_JIT_LEVEL=1 $(BH_PYTHON) $(BH_SRC)/test/python/numpytest.py $(TEST_EXCLUDE)


docs: ## Generate Doxygen documentation
	@echo "${BLUE}Generating Doxygen documentation${NC}"
	mkdir -p $(CPU)/doc/output
	cd $(CPU) && doxygen doc/cpu_engine.doxy
