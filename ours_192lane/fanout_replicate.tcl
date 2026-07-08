# B4: targeted high-fanout replication for the HLS-generated packet-dispatch
# select signal ("sel_rd") implicated in the worst clkout3_primitive path
# (regslice_both_in_j_next_*_U / flow_control_loop_pipe_sequential_init).
# Local KEEP + max_fanout only on these nets, not a global MAX_FANOUT
# (UG949 recommends against a global setting).
#
# This hook runs in OPT_DESIGN.TCL.PRE, before place_design/route_design.
# Deliberately NOT wrapped in catch: if a command/property here is wrong,
# it should fail immediately (cheap, minutes) rather than silently no-op
# and only surface hours later after place/route has already run.
set sel_rd_nets [get_nets -quiet -hierarchical "*sel_rd*"]
if {[llength $sel_rd_nets] > 0} {
    puts "B4: found [llength $sel_rd_nets] sel_rd net(s), applying KEEP + local max_fanout=8"
    set_property KEEP true $sel_rd_nets
    set_property MAX_FANOUT 8 $sel_rd_nets
} else {
    puts "B4: no sel_rd nets matched at OPT_DESIGN.TCL.PRE -- name may differ before optimization; re-check with get_nets -hierarchical *sel_rd* after opt_design if this stays empty"
}
