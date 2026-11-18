# Fig. 4  Typical per-round trajectory

| coverage_deficit | fixed | calibrate_geometry -> dagger_round -> dagger_round |
| coverage_deficit | rule | regenerate_targeted -> randomize_physics -> randomize_physics |
| coverage_deficit | oracle | regenerate_targeted -> regenerate_targeted -> regenerate_targeted |
| geometry_bias | fixed | calibrate_geometry -> dagger_round -> dagger_round |
| geometry_bias | rule | calibrate_geometry -> regenerate_targeted -> dagger_round |
| geometry_bias | oracle | calibrate_geometry -> regenerate_uniform -> regenerate_uniform |
| mixed_cause | fixed | calibrate_geometry -> dagger_round -> dagger_round |
| mixed_cause | rule | calibrate_geometry -> regenerate_targeted -> dagger_round |
| mixed_cause | oracle | calibrate_geometry -> regenerate_targeted -> regenerate_targeted |
