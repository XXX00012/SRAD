#environment
ROOTFS=${EDGE_COMMON_SW_PATH}/rootfs.ext4
IMAGE=${EDGE_COMMON_SW_PATH}/Image
SYSROOT = ${SYSROOT_PATH}/sysroots/cortexa72-cortexa53-xilinx-linux
SDKTARGETSYSROOT = ${SYSROOT}
BASE_PLATFORM ?= ${PLATFORM_REPO_PATHS}/xilinx_vck190_base_202320_1/xilinx_vck190_base_202320_1.xpfm
SHELL := /bin/bash
.SHELLFLAGS := -e -o pipefail -c

# Makefile input options
TARGET_ORIGIN := $(origin TARGET)
TARGET ?= hw

ifeq ($(strip $(MAKECMDGOALS)),sim)
ifeq ($(TARGET_ORIGIN),undefined)
TARGET := sw_emu
endif
endif

XO_DIR=./$(TARGET).xo_dir
XSA_DIR=./$(TARGET).xsa_dir
PACKAGE_DIR = ./$(TARGET).xclbin_dir
SD_CARD_DIR = ./sd_card
PACKAGE_DATA_DIR = ./$(TARGET).package_data
PACKAGE_SD_DATA_DIR = $(PACKAGE_DATA_DIR)/data

# File names and locations
KERNEL_SRC := ./pl/TopPL.cpp
KERNELS := LoadFpgaV5 StoreFpgaV5
KERNEL_XOS := $(addprefix ./pl/,$(addsuffix .xo,$(KERNELS)))

XSA := ./$(XSA_DIR)/svd.xsa
HOST_SRC = ./ps/host.cpp
HOST = host.exe
RUN_SH = ./run.sh
XRT_INI = ./xrt.ini
XCLBIN = ./$(PACKAGE_DIR)/svd.xclbin
GRAPH := ./aie/TopGraph.cpp
DATA_GEN := ./data/gen_case.py
PYTHON := python3
DATA_TAG := 4000x4000
BOARD_INPUT := ./data/input_$(DATA_TAG).txt
BOARD_LAMBDA := ./data/lambda.txt
DATA_FILES := \
	$(BOARD_INPUT) \
	$(BOARD_LAMBDA)
SIM_RUNTIME_FILES := \
	./data/plio_fpga_params.txt \
	./data/plio_fpga_compute.txt \
	./data/q0sqr_ref.txt \
	./data/golden_$(DATA_TAG).txt
DATA_RUNTIME_FILES := \
	$(DATA_FILES) \
	$(SIM_RUNTIME_FILES)
PROJECT_HEADERS := \
	./aie/TopGraph.h \
	./aie/Config.h \
	./aie/ProcessGraph/StencilCoreGraph.h \
	./aie/ProcessUnit/include.h \
	./aie/ProcessUnit/srad.h
AIE_TARGET_STAMP := .aie_target.$(TARGET)
AIE_DEPS := Makefile \
	$(AIE_TARGET_STAMP) \
	$(GRAPH) \
	$(PROJECT_HEADERS) \
	./aie/ProcessGraph/StencilCoreGraph.cpp \
	./aie/ProcessUnit/srad_fpga.cc
HOST_DEPS := Makefile \
	$(HOST_SRC) \
	$(PROJECT_HEADERS)

GRAPH_O := libadf.a
CONFIG_FILE := ./conn.cfg

# Command-line options
VPP := v++
AIECC := v++ -c --mode aie
AIESIM := aiesimulator
X86SIM := x86simulator

HW_EMU_CMD := ./launch_hw_emu.sh -g

# Frequency Option
AIE_FREQ = --freqhz=450000000

# compile aie
AIE_INCLUDE_FLAGS := --include "$(XILINX_VITIS)/aietools/include" --include "./aie" --include "./"
AIE_STRATEGY := --aie.xlopt=0 --aie.runtime-opt --aie.verbose --aie.Xchess=main:backend.mist2.xargs=-ggraph
AIE_FLAGS := $(AIE_INCLUDE_FLAGS) $(AIE_STRATEGY) --platform $(BASE_PLATFORM) --work_dir ./Work
AIE_FLAGS += $(AIE_FREQ)

ifeq ($(TARGET),sw_emu)
	AIE_FLAGS += --target x86sim
else
	AIE_FLAGS += --target hw
endif

# kernel build config
VPP_XO_FLAGS := -c -t $(TARGET) --platform $(BASE_PLATFORM) --save-temps -g
VPP_LINK_FLAGS := -l -t $(TARGET) --platform $(BASE_PLATFORM) $(GRAPH_O) $(KERNEL_XOS) --save-temps -g --config $(CONFIG_FILE) -o $(XSA)
VPP_LINK_FLAGS += --vivado.impl.jobs 256

CXX := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++

GCC_FLAGS := -Wall -c \
			 -std=c++20 \
			 -Wno-int-to-pointer-cast \
			 --sysroot=$(SYSROOT)

GCC_INCLUDES := -I$(SYSROOT)/usr/include/xrt \
				-I./ \
				-I${XILINX_VITIS}/aietools/include \
				-I${XILINX_VITIS}/include \
				-I${XILINX_HLS}/include \
				-I./aie

GCC_LIB := -lxaiengine -ladf_api_xrt -lxrt_core -lxrt_coreutil \
		   -L$(SYSROOT)/usr/lib \
		   --sysroot=$(SYSROOT) \
		   -L${XILINX_VITIS}/aietools/lib/aarch64.o

.ONESHELL:
.PHONY: clean all kernels aie sim xsa host package package_data sd_card run run_emu data data_runtime \
	clear_aie clear_pl clear_ps clear_xsa clear_package clear_all

all: aie kernels xsa host package sd_card

sd_card: package
	rm -rf $(SD_CARD_DIR)
	mkdir -p $(SD_CARD_DIR)/data
	cp $(HOST) $(SD_CARD_DIR)/
	cp $(XCLBIN) $(SD_CARD_DIR)/svd.xclbin
	cp $(RUN_SH) $(SD_CARD_DIR)/
	cp $(XRT_INI) $(SD_CARD_DIR)/
	cp $(DATA_FILES) $(SD_CARD_DIR)/data/
	chmod +x $(SD_CARD_DIR)/run.sh
	@echo "COMPLETE: board staging files refreshed in $(SD_CARD_DIR)."

clean: clear_all

data: $(DATA_FILES)
data_runtime: $(DATA_RUNTIME_FILES)

$(DATA_FILES): $(DATA_GEN) ./aie/Config.h
	$(PYTHON) $(DATA_GEN)

$(SIM_RUNTIME_FILES): $(DATA_GEN) ./aie/Config.h
	$(PYTHON) $(DATA_GEN) --sim-runtime

package_data:
	rm -rf $(PACKAGE_DATA_DIR)
	mkdir -p $(PACKAGE_SD_DATA_DIR)
	cp $(DATA_FILES) $(PACKAGE_SD_DATA_DIR)/

######################################################
# This step compiles the AIE graph and HLS kernels.

aie: $(GRAPH_O)

kernels: $(KERNEL_XOS)

./pl/%.xo: $(KERNEL_SRC) ./aie/Config.h
	mkdir -p $(XO_DIR)
	$(VPP) $(VPP_XO_FLAGS) -k $* -o $@ $(KERNEL_SRC) | tee $(XO_DIR)/$*.log

sim: data_runtime $(GRAPH_O)
ifeq ($(TARGET),sw_emu)
	$(X86SIM) --pkg-dir=./Work
else
	$(AIESIM) --pkg-dir=./Work
endif

$(AIE_TARGET_STAMP):
	rm -rf Work $(GRAPH_O)
	rm -f .aie_target.*
	touch $(AIE_TARGET_STAMP)

$(GRAPH_O): $(AIE_DEPS)
	$(AIECC) $(AIE_FLAGS) $(GRAPH)

########################################################
xsa: $(XSA)
$(XSA): $(GRAPH_O) $(KERNEL_XOS) $(CONFIG_FILE) Makefile
	rm -rf _x $(XSA_DIR)
	mkdir -p $(XSA_DIR)
	$(VPP) $(VPP_LINK_FLAGS) || (echo "task: [xsa] failed error code: $$?"; exit 1)
	@echo "COMPLETE: .xsa created."

########################################################
host: $(HOST)

$(HOST): $(HOST_DEPS) $(GRAPH_O)
	$(CXX) $(GCC_FLAGS) $(GCC_INCLUDES) -c $(HOST_SRC) -o host.o
	$(CXX) $(GCC_FLAGS) $(GCC_INCLUDES) -c ./Work/ps/c_rts/aie_control_xrt.cpp -o aie_control_xrt.o
	$(CXX) host.o aie_control_xrt.o $(GRAPH_O) $(GCC_LIB) -std=c++20 -o $(HOST)
	@echo "COMPLETE: Host application created."

########################################################
package: package_data
	@for f in $(XSA) $(GRAPH_O) $(HOST) $(RUN_SH) $(XRT_INI); do \
		if [ ! -f "$$f" ]; then \
			echo "ERROR: missing $$f. Build required artifacts before make package."; \
			exit 1; \
		fi; \
	done
	mkdir -p $(PACKAGE_DIR)
	$(VPP) -p -t ${TARGET} \
		-f ${BASE_PLATFORM} \
		--package.out_dir=$(PACKAGE_DIR) \
		--package.rootfs=${ROOTFS} \
		--package.image_format=ext4 \
		--package.boot_mode=sd \
		--package.kernel_image=${IMAGE} \
		--package.defer_aie_run \
		--package.sd_file $(HOST) \
		--package.sd_file $(RUN_SH) \
		--package.sd_file $(XRT_INI) \
		--package.sd_dir $(PACKAGE_SD_DATA_DIR) \
		$(XSA) $(GRAPH_O) -o $(XCLBIN)
	@echo "COMPLETE: hardware package created."

run: all run_emu

run_emu: package
ifeq (${TARGET},hw_emu)
	cd ./$(PACKAGE_DIR)
	$(HW_EMU_CMD)
else
	@echo "Hardware build, no emulation executed."
endif

clear_aie:
	rm -rf Work *.a
	rm -f .aie_target.*
clear_pl:
	rm -rf $(XO_DIR) pl/*.xo
clear_ps:
	rm -rf *.o *.exe
clear_xsa:
	rm -rf $(XSA_DIR) _x
clear_package:
	rm -rf $(PACKAGE_DIR) $(SD_CARD_DIR) $(PACKAGE_DATA_DIR)
clear_all: clear_aie clear_pl clear_ps clear_xsa clear_package
	rm -rf hls TopPL logs
	rm -rf _x v++* .Xil *.db *.log *.wcfg *.wdb pl_sample_counts *.csv .AIE_* ISS_* aiesimulator_* x86simulator_* function_*
