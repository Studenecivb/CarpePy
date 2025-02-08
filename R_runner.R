# Welcome to CarpePy used in R
# First install or load the reticulate R package
# install.packages('reticulate')
library(reticulate)

# Not necessary usually - in case there is any issue with virtualenv do these steps
virtualenv_create("r-reticulate")
use_virtualenv("r-reticulate", required = TRUE)

# Install carpepy and load it
py_install("carpepy")
carpepy <- import("carpepy")

# This example is running a python code fully, this is not necessary, it is possible to load the data
# and formatting it yourself in R and then running the functions from Python. For more information on how
# to do this please read the reticulate documentation: https://rstudio.github.io/reticulate/articles/package.html
# And to see how the input data of the particular carpepy functions should look, please checkout the manual.
# This example shows Pneumocystis processing from the article: ADD

# Read the CSV file into Python
py_run_string("import pandas as pd")
py_run_string("from carpepy import DiemPlotPrep, diemUnitPlot, diemIrisPlot, RichRLE")
py_run_string("import numpy as np")
py_run_string("HonzaPneumo_df = pd.read_csv('your_input_data.csv', sep=',')")

# Now we do a few steps of formatting the input data to match the format described in the manual:
py_run_string('
third_column = [row[2] for row in HonzaPneumo_df.values.tolist()]
fourth_column = [row[3] for row in HonzaPneumo_df.values.tolist()]
HonzaPneumoBED = list(zip(third_column, fourth_column))
')

py_run_string('
first_row = HonzaPneumo_df.columns.tolist()
HonzaPneumoIndIDs = first_row[18:-6]
HonzaPneumoNinds = len(HonzaPneumoIndIDs)
')

py_run_string('
column_indices = list(range(18, 18 + len(HonzaPneumoIndIDs)))
HonzaPneumoSelected = [[row[i] for i in column_indices] for row in HonzaPneumo_df.values.tolist()]
HonzaPneumoMarkers = ["".join(map(str, row)) for row in HonzaPneumoSelected]
')

py_run_string('
HonzaPneumoPolariseNjoin = [list(row) + [marker] for row, marker in zip(HonzaPneumoBED, HonzaPneumoMarkers)]
')

# Run diemplotpre, here we are not filtering by diagnostic index
py_run_string("
plot_theme = 'Pneumocystis'

PneumoPlotPrep = DiemPlotPrep(
    plot_theme='Pneumocystis',
    polarised_data=HonzaPneumoPolariseNjoin,
    ind_ids=HonzaPneumoIndIDs,
    di_threshold='NO DI FILTER',
    di_column=5,
    phys_res=1,
    ticks='kb'
)

PneumoPlotPrep.format_bed_data()
")


# Run the unit plots for each chromosome
py_run_string("
for i in range(min(len(PneumoPlotPrep.unit_plot_prep), len(PneumoPlotPrep.DIfilteredBED_formatted))):
    diemUnitPlot(
        PneumoPlotPrep.unit_plot_prep[i], 
        bed_data=PneumoPlotPrep.DIfilteredBED_formatted[i],
        index=i+1,
        path='your_output_path',
        names_list=PneumoPlotPrep.IndIDs_ordered,
        ticks='kb'
    )
")

# Run the iris plot for the whole genome
py_run_string("
heatmap_pre_values = list(HonzaPneumo_df.iloc[:, -4])
rle_heatmap_values = np.array(RichRLE(heatmap_pre_values)).T
heatmap_map = np.delete(rle_heatmap_values, 1, axis=1)

diemIrisPlot(PneumoPlotPrep.diemDITgenomes_ordered, names=PneumoPlotPrep.IndIDs_ordered,
         bed_info=PneumoPlotPrep.iris_plot_prep,
         length_of_chromosomes=PneumoPlotPrep.length_of_chromosomes,
         heatmap=heatmap_map,
         path='your_output_path',
         png='cute_iris_lot')")




