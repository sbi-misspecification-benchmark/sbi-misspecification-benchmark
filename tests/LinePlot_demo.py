from src.utils.LinePlot import LinePlot

# Plot 1
plot1 = LinePlot(
    data_sources="dummy_csv_2x2_grid_default.csv",
    title="Dummy 2x2 Benchmark Grid"
)

# Plot 2
plot2 = LinePlot(
    data_sources="dummy_csv_4x4_grid_linear.csv",
    log_x=False,
    title="Dummy 4x4 Benchmark Grid"
)

# Plot 3
plot3 = LinePlot(
    data_sources="dummy_csv_6x8_grid_multimetrics.csv",
    title="Dummy 6x8 Benchmark Grid",
    row_order=["Task 3", "Task 2", "Task 1"],
    col_order=["Method 8", "Method 7", "Method 6", "Method 5", "Method 4", "Method 3", "Method 2", "Method 1"]
)

# Plot 4
plot4 = LinePlot(
    x="tau_m",
    data_sources="dummy_csv_4x4_grid_with_tau.csv",
    title="Dummy 4x4 Benchmark Grid (C2ST vs Tau)",
    log_x=False
    )

# Create the plots
plot1.run()
plot2.run()
plot3.run()
plot4.run()
