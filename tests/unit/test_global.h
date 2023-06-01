// Should already be imported by test_main.cu

using namespace pyroclastmpm;

// Functions to test
// [ ] set_global_dt
// [ ] set_global_shapefunction (need to refactor this)
// [ ] set_global_output_dir
// [ ] set_globals
// [ ] set_global_step

// Not I get wierd linking errors with external globla memory when trying to
// compile these tests. A solution is to use getter functions . . . need to see
// how this can be done

TEST(Global, SET_GLOBAL_DT) {
  Real dt = 0.1;
  set_global_dt(dt);
  // EXPECT_EQ(dt_cpu,dt );
}