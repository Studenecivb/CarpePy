# CarpePy
Welcome to the CarpePy documentation!

CarpePy is a toolset for visualising polarised genomic data. It is
dependent on a pre-processed data from *diem* package: https://github.com/StuartJEBaird/diem

CarpePy is dependent on pandas and numpy Python packages and we recommend running it using a virtual environment.

### Example of running CarpePy with Pneumocystis data
Pneumocystis data: *Jan Petružela, Beate Nürnberger, Alexis Ribas, et al. Comparative genomic analysis of co-occurring hybrid zones of house mouse parasites Pneumocystis murina and Syphacia obvelata using genome polarisation. Authorea. January 31, 2025.*

First load the input data from *diem* as a pandas dataframe:
```python
HonzaPneumo_df = pd.read_csv(file_path,sep=',')
HonzaPneumo = HonzaPneumo_df.values.tolist()
```
This is how we want the HonzaPneumo_df to look like approximately:

| Row | diem_markpos | scaffold | refpos         | V3  | ... | SK1151_DM | SU4201_DM | ...  | i_sig | o_sig | admixture_category | genic | cds | msg |
|-----|--------------|----------|----------------|-----|-----|-----------|-----------|------|-------|-------|--------------------|-------|-----|-----|
| 0   | 0            | m1       | AFWA02000001.1 | 44  | ... | 0         | 2         | .... | 0     | 0     | barr               | 0     | 0   | 0   |
| 1   | 1            | m2       | AFWA02000001.1 | 94  | ... | 0         | _         | ...  | 0     | 0     | barr               | 0     | 0   | 0   |
| 2   | 2            | m3       | AFWA02000001.1 | 314 | ... | 0         | 2         | ...  | 0     | 0     | barr               | 0     | 0   | 0   |

Now we are going to take the thrid and fourth column to get the BED information into a separate variable:
```python
third_column = [row[2] for row in HonzaPneumo]
fourth_column = [row[3] for row in HonzaPneumo]
HonzaPneumoBED = list(zip(third_column, fourth_column))
```
As the next step, we want to extract the names of the individuals:
```python
first_row = HonzaPneumo_df.columns.tolist()
HonzaPneumoIndIDs = first_row[18:-6]
```
And then the markers and the selected input data:
```python
column_indices = list(range(18, 18 + len(HonzaPneumoIndIDs)))
HonzaPneumoSelected = [[row[i] for i in column_indices] for row in HonzaPneumo]
HonzaPneumoMarkers = ["".join(map(str, row)) for row in HonzaPneumoSelected]
HonzaPneumoPolariseNjoin = [list(row) + [marker] for row, marker in zip(HonzaPneumoBED, HonzaPneumoMarkers)]
```
Now we are finally ready to run the diem Plot Prepper:
```python
plot_theme = "Pneumocystis"

PneumoPlotPrep = DiemPlotPrep(plot_theme='Pneumocystis', polarised_data=HonzaPneumoPolariseNjoin, ind_ids=HonzaPneumoIndIDs,
                         di_threshold="NO DI FILTER", di_column=5, phys_res=1, ticks='kb')
PneumoPlotPrep.format_bed_data()
```
The arguments include the plot theme, the polarised and processed data, the index names, diagnostic index filtering
if we want any and the column we want to use for it, resolution (in this case 1) and lastly
the tick sizes we want - either kb or mb, depending on our data.

Now we are all prepped to run either the Unit plots - which represent the unit of our genome, either 
a scaffold or chromosome or then the IrisPlot which shows us the whole genome.
```python
for i in range(len(PneumoPlotPrep.unit_plot_prep)):
    diemUnitPlot(PneumoPlotPrep.unit_plot_prep[i], bed_data=PneumoPlotPrep.DIfilteredBED_formatted[i],
                        index=i+1,
                      path='output_path',
                      names_list=PneumoPlotPrep.IndIDs_ordered, ticks='kb')
```
The output of the unit plot:
![unit_plot.png](assets%2Funit_plot.png)

And now the IrisPlot:
```python


diemIrisPlot(PneumoPlotPrep.diemDITgenomes_ordered, names=PneumoPlotPrep.IndIDs_ordered,
         bed_info=PneumoPlotPrep.iris_plot_prep,
         length_of_chromosomes=PneumoPlotPrep.length_of_chromosomes,
         heatmap=heatmap_map,
         path=output_path,
         png='cute_iris', pdf='cute_iris')
```
The arguments include the chromosome names, BED information and the diemDITheredgenomes that are ordered according to the Hybrid Index.
If you do not add any png or pdf name, the plot will just be shown, pdf and png names allow it to be saved into a folder.

We can also add a heatmap to the IrisPlot and it should be processed:
```python
heatmap_pre_values = list(HonzaPneumo_df.iloc[:, -4])
rle_heatmap_values = np.array(RichRLE(heatmap_pre_values)).T
heatmap_map = np.delete(rle_heatmap_values, 1, axis=1)
```
The output of Iris Plot:
![iris_plot.png](assets%2Firis_plot.png)

Please if you have any questions, contact us on: ninahaladova@gmail.com

Cite as: Baird, S. J. E., & Daley, N. (2025). CarpePy (Version 0.0.1) [Computer software]

