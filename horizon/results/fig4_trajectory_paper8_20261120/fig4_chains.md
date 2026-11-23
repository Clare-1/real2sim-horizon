# Fig. 4  Typical per-round trajectory

| coverage_deficit | fixed | calibrate_geometry -> dagger_round -> dagger_round |
| coverage_deficit | rule | regenerate_targeted -> stop |
| coverage_deficit | oracle | regenerate_targeted -> regenerate_targeted -> regenerate_targeted |
| geometry_bias | fixed | calibrate_geometry -> dagger_round -> dagger_round |
| geometry_bias | rule | calibrate_geometry -> regenerate_targeted -> stop |
| geometry_bias | oracle | calibrate_geometry -> regenerate_uniform -> regenerate_uniform |
| healthy | fixed | calibrate_geometry -> dagger_round -> dagger_round |
| healthy | rule | stop |
| healthy | oracle | stop |
| recovery_deficit | fixed | calibrate_geometry -> dagger_round -> dagger_round |
| recovery_deficit | rule | dagger_round -> dagger_round -> dagger_round |
| recovery_deficit | oracle | add_recovery_data -> add_recovery_data -> add_recovery_data |
