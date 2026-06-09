.DEFAULT_GOAL := all

.PHONY: all sd_card clean clear_all clear_aie clear_pl clear_ps clear_xsa clear_package \
	aie sim xsa host package run run_emu

all sd_card clean clear_all clear_aie clear_pl clear_ps clear_xsa clear_package \
aie sim xsa host package run run_emu:
	$(MAKE) -C .. $@

%:
	$(MAKE) -C .. $@
