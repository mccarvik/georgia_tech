#h# ***** Makefile to build Macsim and run benchmarks *****

# .SILENT:

# Environment check
ifeq (${MACSIM_DIR},)
    $(error "'MACSIM_DIR' environment variable not set; did you forget to source the sourceme script?")
endif

CC        := g++
LD        := g++
CXXFLAGS  := -Wall -Wno-unused-variable -Wno-unused-but-set-variable -std=c++20 -I src/macsim
LDFLAGS   := -lz

ifeq ($(DEBUG),1)
    CXXFLAGS += -g -O0
else
    CXXFLAGS += -O3
endif

ifeq ($(COV),1)
    CXXFLAGS+= -fPIC -fprofile-arcs -ftest-coverage
endif

MODULES   := macsim exec utils
SRC_DIR   := $(addprefix src/,$(MODULES)) src
BUILD_DIR := $(addprefix build/,$(MODULES)) build

SRC       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
SRC       := src/main.cpp $(SRC)
OBJ       := $(patsubst src/%.cpp,build/%.o,$(SRC))
DEP       := $(OBJ:.o=.d)
INCLUDES  := $(addprefix -I,$(SRC_DIR))

vpath %.cpp $(SRC_DIR)

define make-goal
$1/%.o: %.cpp
	@printf "> Compiling    $$(@F)\n"
	$(CC) $(CXXFLAGS) $(INCLUDES) -MMD -c $$< -o $$@
endef

OUTPUT_TAR  :=  CS8803proj_combined_${USER}.tar
LOG_DIR     :=  ${MACSIM_DIR}/log
TRACE_DIR   :=  ${MACSIM_TRACE_DIR}

$(shell mkdir -p $(LOG_DIR))

.PHONY: all checkdirs clean

all: checkdirs macsim                       #t# Build macsim

macsim: $(OBJ)
	@printf "> Linking  $(@F)\n"
	$(LD) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

.PHONY: cov
cov: $(SRC)                                 #t# generate coverage metadata
	gcov $^ -o build

.PHONY: traces
traces:                                     #t# Download traces (if not present)
	bash ./scripts/get_traces.sh $(TRACE_DIR)

# -----------------------------------------------------------------------------------------
# Run Tasks (Combined Project 3 + Project 4)
# -----------------------------------------------------------------------------------------

.PHONY: task1
task1: traces                               #t# Run Task 1: Compute Core (Naive)
	# Assumes 'CC' config corresponds to Compute Core logic in xmls
	python3 scripts/runner.py -c CC -b lavaMD_5:gemm_float:gemm_half:cnn_float:cnn_half:gpt2_float:gpt2_half -t $(TRACE_DIR) -l $(LOG_DIR) -r -d $(LOG_DIR)/results.txt -j $(LOG_DIR)/results.json

.PHONY: task2
task2: traces                               #t# Run Task 2: Tensor Core & Dependency Checks
	# Assumes 'CC:TC' config corresponds to Tensor Core logic in xmls
	python3 scripts/runner.py -c CC:TC -b lavaMD_5:gemm_float:gemm_half:cnn_float:cnn_half:gpt2_float:gpt2_half -t $(TRACE_DIR) -l $(LOG_DIR) -r -d $(LOG_DIR)/results.txt -j $(LOG_DIR)/results.json

.PHONY: task3
task3: traces                               #t# Run Task 3: GTO Scheduler
	python3 scripts/runner.py -c RR:GTO -b lavaMD_5:gemm_float:gemm_half:cnn_float:cnn_half:gpt2_float:gpt2_half -t $(TRACE_DIR) -l $(LOG_DIR) -r -d $(LOG_DIR)/results.txt -j $(LOG_DIR)/results.json

.PHONY: task4
task4: traces                               #t# Run Task 4: CCWS Scheduler
	python3 scripts/runner.py -c RR:GTO:CCWS -b lavaMD_5:gemm_float:gemm_half:cnn_float:cnn_half:gpt2_float:gpt2_half -t $(TRACE_DIR) -l $(LOG_DIR) -r -d $(LOG_DIR)/results.txt -j $(LOG_DIR)/results.json

.PHONY: run-all-tasks
run-all-tasks: traces                            #t# Run all configs and export combined results for submission
	python3 scripts/runner.py -c CC:TC:RR:GTO:CCWS -b lavaMD_5:gemm_float:gemm_half:cnn_float:cnn_half:gpt2_float:gpt2_half -t $(TRACE_DIR) -l $(LOG_DIR) -r -d $(LOG_DIR)/results.txt -j $(LOG_DIR)/results.json


.PHONY: plot
plot:                                       #t# Plot stats for report
	python3 scripts/plot.py -t $(TRACE_DIR) -l $(LOG_DIR) $(LOG_DIR)/results.json NUM_STALL_CYCLES
	python3 scripts/plot.py -t $(TRACE_DIR) -l $(LOG_DIR) $(LOG_DIR)/results.json MISSES_PER_1000_INSTR
	python3 scripts/plot.py -t $(TRACE_DIR) -l $(LOG_DIR) $(LOG_DIR)/results.json NUM_CYCLES
	python3 scripts/plot.py -t $(TRACE_DIR) -l $(LOG_DIR) $(LOG_DIR)/results.json NUM_STALL_CYCLES

# -----------------------------------------------------------------------------------------

.PHONY: submit
submit: clean                               #t# Generate submission tarball
	@echo "Generating tar..."
	tar --exclude=.python3 \
	--exclude=.vscode \
	--exclude=build \
	--exclude=log \
	--exclude=*.zip \
	--exclude=*.tar* \
	--exclude=*.png \
	--exclude=*__pycache__* \
	--exclude=macsim_traces \
	-czhvf $(OUTPUT_TAR) *
	

.PHONY: clean
clean:                                      #t# Clean build files
	rm -rf $(BUILD_DIR)
	rm -f macsim
	rm -f *.log
	rm -f $(OUTPUT_TAR)
	rm -f *.gcov

.PHONY: clean-logs
clean-logs:                                 #t# Clean logs (log dir)
	rm -rf $(LOG_DIR)/*

.PHONY: clean-all
clean-all: clean clean-logs                 #t# Clean build files and logs

.PHONY: help
help: Makefile                              #t# Print help
# Print Header comments '#h#'
	@sed -n 's/^#h# //p' $<

# Print Target descriptions '#t#'
	@printf "\nTARGETs:\n"
	@grep -E -h '\s#t#\s' $< | awk 'BEGIN {FS = ":.*?#t# "}; {printf "\t%-20s %s\n", $$1, $$2}'

# Print footer '#f#'
	@sed -n 's/^#f# //p' $<

-include $(DEP)
$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))
