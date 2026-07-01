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

# Bare `make sim` is a quick correctness check; request TARGET=hw explicitly
# when instruction-level AIE simulation is needed.
ifeq ($(strip $(MAKECMDGOALS)),sim)
ifeq ($(TARGET_ORIGIN),undefined)
TARGET := sw_emu
endif
endif

XO_DIR=./$(TARGET).xo_dir
XSA_DIR=./$(TARGET).xsa_dir
PACKAGE_DIR = ./$(TARGET).xclbin_dir
SD_CARD_DIR = ./sd_card

# File names and locations
PL_KERNEL := TopPL
PL_KERNEL_SRC := ./pl/$(PL_KERNEL).cpp
PL_KERNEL_XO := ./pl/$(PL_KERNEL).xo
KERNEL_XOS := $(PL_KERNEL_XO)

XSA := ./$(XSA_DIR)/svd.xsa
HOST_SRC = ./ps/host.cpp
HOST = host.exe
RUN_SH = ./run.sh
XRT_INI = ./xrt.ini
XCLBIN = ./$(PACKAGE_DIR)/svd.xclbin
GRAPH := ./aie/TopGraph.cpp
DATA_FILES := \
	./data/input_image.txt
SIM_DATA_FILES := \
	./data/input_image_sim.txt
DATA_GEN := ./data/gen_case.py
DATA_VERIFY := ./data/verify_srad.py
PLIO_DATA_FILES := ./data/plio_ours_j.txt
SIM_GENERATED_DATA_FILES := \
	$(SIM_DATA_FILES) \
	$(PLIO_DATA_FILES)
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
	./aie/ProcessUnit/srad_local_q.cc \
	./aie/ProcessUnit/srad_coeff_update.cc \
	./aie/ProcessUnit/srad_row_stats.cc
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
SIM_OUTPUT := ./data/aiesim_j_next.txt
SIM_OUTPUT_DIRS := ./x86simulator_output ./aiesimulator_output ./Work/x86simulator_output ./Work/aiesimulator_output

HW_EMU_CMD := ./launch_hw_emu.sh -g
#HW_EMU_CMD := ./launch_hw_emu.sh -add-env AIE_COMPILER_WORKDIR=../Work -aie-sim-options ../aiesimulator_output/aiesim_options.txt

# Frequency Option
AIE_FREQ = --freqhz=450000000

# compile aie
AIE_INCLUDE_FLAGS := --include "$(XILINX_VITIS)/aietools/include" --include "./aie" --include "./"
AIE_STRATEGY := --aie.runtime-opt
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

VPP_FLAGS := $(VPP_LINK_FLAGS)

CXX := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++

GCC_FLAGS := -Wall -c \
			 -std=c++20 \
			 -Wno-int-to-pointer-cast \
			 -pthread \
			 --sysroot=$(SYSROOT)

GCC_INCLUDES := -I$(SYSROOT)/usr/include/xrt \
				-I./ \
				-I${XILINX_VITIS}/aietools/include \
				-I${XILINX_VITIS}/include \
				-I${XILINX_HLS}/include \
				-I./aie

GCC_LIB := -lxaiengine -ladf_api_xrt -lxrt_core -lxrt_coreutil \
		   -pthread \
		   -L$(SYSROOT)/usr/lib \
		   --sysroot=$(SYSROOT) \
		   -L${XILINX_VITIS}/aietools/lib/aarch64.o

LDCLFLAGS := $(GCC_LIB)

.ONESHELL:
.PHONY: clean all kernels aie sim sim_fast xsa host package sd_card run run_emu data \
	force_host \
	clear_aie clear_pl clear_ps clear_xsa clear_package clear_all

all: aie kernels xsa host package sd_card
sd_card: $(XCLBIN) $(HOST) $(RUN_SH) $(XRT_INI) $(DATA_FILES)
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
data:
	python $(DATA_GEN)

$(DATA_FILES): $(DATA_GEN) ./aie/Config.h
	python $(DATA_GEN)

$(SIM_GENERATED_DATA_FILES): $(DATA_GEN) ./aie/Config.h
	python $(DATA_GEN) --sim

######################################################
# This step compiles the HLS C kernels and creates the *.xo's
# which is used as the output and from the *.cpp files.

aie: $(GRAPH_O)

kernels: $(KERNEL_XOS)

$(PL_KERNEL_XO): $(PL_KERNEL_SRC) ./aie/Config.h ./aie/ProcessUnit/include.h
	mkdir -p $(XO_DIR)
	$(VPP) $(VPP_XO_FLAGS) -k $(PL_KERNEL) -o $(PL_KERNEL_XO) $(PL_KERNEL_SRC) | tee $(XO_DIR)/$(PL_KERNEL).log

#AIE or X86 Simulation
sim: $(GRAPH_O) $(PLIO_DATA_FILES)
	rm -f $(SIM_OUTPUT)
	rm -rf $(SIM_OUTPUT_DIRS)

ifeq ($(TARGET),sw_emu)
	@echo "SIM: TARGET=sw_emu, running x86simulator for quick correctness."
	$(X86SIM) --pkg-dir=./Work
else
	@echo "SIM: TARGET=$(TARGET), running aiesimulator."
	$(AIESIM) --pkg-dir=./Work
endif
	python $(DATA_VERIFY) --compare

sim_fast:
	$(MAKE) clear_aie
	$(MAKE) sim TARGET=sw_emu

#AIE or X86 compilation
$(AIE_TARGET_STAMP):
	rm -rf Work $(GRAPH_O)
	rm -f .aie_target.*
	touch $(AIE_TARGET_STAMP)

$(GRAPH_O): $(AIE_DEPS)
	$(AIECC) $(AIE_FLAGS) $(GRAPH)
#####################################################

########################################################
# Once the kernels and graph are generated, you can build
# the hardware part of the design. This creates an xsa
# that will be used to run the design on the platform.
xsa: $(XSA)
$(XSA): $(GRAPH_O) $(KERNEL_XOS) $(CONFIG_FILE) Makefile
	rm -rf _x $(XSA_DIR)
	rm -f ./pl/Q0Ctrl.xo
	mkdir -p $(XSA_DIR)
	$(VPP) $(VPP_LINK_FLAGS) || (echo "task: [xsa] failed error code: $$?"; exit 1)
	@echo "COMPLETE: .xsa created."
########################################################

############################################################################################################################
# For sw emulation, hw emulation and hardware, compile the PS code and generate the host.exe. This is needed for creating the sd_card.

host: $(HOST)

$(HOST): force_host $(HOST_DEPS) $(GRAPH_O)
	$(CXX) $(GCC_FLAGS) $(GCC_INCLUDES) -c $(HOST_SRC) -o host.o
	$(CXX) $(GCC_FLAGS) $(GCC_INCLUDES) -c ./Work/ps/c_rts/aie_control_xrt.cpp -o aie_control_xrt.o
	$(CXX) host.o aie_control_xrt.o $(GRAPH_O) $(GCC_LIB) -std=c++20 -o $(HOST)
	@echo "COMPLETE: Host application created."

############################################################################################################################

##################################################################################################
# Depending on the TARGET, it'll either generate the PDI for sw_emu, hw_emu or hw.

package: $(XCLBIN)
$(XCLBIN): $(XSA) $(GRAPH_O) $(HOST) $(RUN_SH) $(XRT_INI) $(DATA_FILES) Makefile
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
		--package.sd_dir ./data \
		$(XSA) $(GRAPH_O) -o $(XCLBIN)
	@echo "COMPLETE: emulation package created."

###################################################################################################

#Build the design and then run sw/hw emulation
run: all run_emu

###########################################################################
run_emu: $(XCLBIN)
# If the target is for HW_EMU, launch the emulator
ifeq (${TARGET},hw_emu)
	cd ./$(PACKAGE_DIR)
	$(HW_EMU_CMD)
else
	@echo "Hardware build, no emulation executed."
endif

###########################################################################

clear_aie:
	rm -rf Work *.a
	rm -f .aie_target.*
clear_pl:
	rm -rf $(XO_DIR) pl/*.xo
	rm -f ./pl/Q0Ctrl.xo
clear_ps:
	rm -rf *.o *.exe
clear_xsa:
	rm -rf $(XSA_DIR) _x
clear_package:
	rm -rf $(PACKAGE_DIR) $(SD_CARD_DIR)
clear_all: clear_aie clear_pl clear_ps clear_xsa clear_package
	rm -rf hls TopPL Q0Ctrl logs
	rm -rf _x v++* .Xil *.db *.log *.wcfg *.wdb pl_sample_counts *.csv .AIE_* ISS_* aiesimulator_* x86simulator_* function_*
