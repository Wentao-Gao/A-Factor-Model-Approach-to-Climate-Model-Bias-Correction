# A-deconfounding-approach-to-climate-model-bias-correction

This code is for AAAI submission paper: A-deconfounding-approach-to-climate-model-bias-correction

Our study area is South Australia.



<img src="figures/Study_area_NCEP_final.png" alt="Figure" width="70%">



As descriped in the paper, our method is devided into two part, 'Deconfounding' and 'Correction'

![Figure1](figures/Process_final.png)


## Data Preparation

### Simulation Data

To generate simulation data for Deconfounding, follow these steps:

1. Run the following command to generate the simulated datasets:

    ```python
    python main_run_simulation.py
    ```

   This will produce the simulated datasets from two sources and combine them into a CSV file. After the training process, a `.pkl` file will be generated.

2. To process the results and convert them into a CSV file, run:

    ```python
    python result_process.py 
    ```

### Real-World Data

For real-world data, download the necessary datasets from the following sources:

- **IPSL Data Portal:** [IPSL CMIP6 Data](https://aims2.llnl.gov/search/cmip6)
- **NCEP-NCAR Reanalysis 1 Data Portal:** [NCEP-NCAR Reanalysis](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html)

For both datasets, select all variables at the surface level and at 2 meters, and download the monthly data. The data will be in `.nc` (NetCDF) format. Convert these `.nc` files to CSV format, and extract the relevant data for South Australia.

**Note:** For IPSL data, you'll need to download all experimental settings, ranging from `r1p1i1f1` to `r33p1i1f1`.

After obtaining the South Australia CSV file, remove any columns with a significant amount of missing data.



## Deconfounding

First part is most import part as it is the first time to bring some insight from deconfounding in causal inference to climate bias correction
which do not need to assume "All variable are observed". We gonna show our code with the simulation dataset. The simulation dataset is created 
based on the summary causal graph as Figure(c) below shows.


![Figure2](figures/Summary%20causal%20graph_final.png)

As for case study real world dataset, can be download from the link below:

- IPSL data portal: https://aims2.llnl.gov/search/cmip6

- NCEP-NCAR reanalysis 1 data portal: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html

![factor model](figures/factor%20model.png)
  

## Correcrtion

Second part is correction which is trying to used the latent confounder learnt by deconfounding part as an additional feature for precipitation 
correction. As descriped in the paper, we choose the SOTA model iTransformer to perform this step. Implementation is given in https://github.com/thuml/iTransformer.

