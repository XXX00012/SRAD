# B1: diagnostic pass after the final post-route phys_opt_design step --
# capture Vivado's own QoR suggestions for the clkout3_primitive timing
# failures. Read-only: does not change placement or routing by itself.
# Wrapped in catch so an unsupported command here cannot abort the build
# that has already gotten this far (route + post-route physopt done).
if {[catch {
    report_qor_suggestions -name qor_run
    write_qor_suggestions -strategies -file qor_suggestions.rqs
} err]} {
    puts "B1: WARNING - report_qor_suggestions/write_qor_suggestions failed ($err), skipping"
}
