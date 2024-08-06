# A-deconfounding-approach-to-climate-model-bias-correction

This code is for AAAI submission paper: A-deconfounding-approach-to-climate-model-bias-correction

As descriped in the paper, our method is devided into two part, 'Deconfounding' and 'Correction'

<img width="1310" alt="Process" src="https://github.com/user-attachments/assets/30942574-8fee-4c72-9e9e-54e71bf7daf1">


## Deconfounding

First part is most import part as it is the first time to bring some insight from deconfounding in causal inference to climate bias correction
which do not need to assume "All variable are observed". We gonna show our code with the simulation dataset. The simulation dataset is created 
based on the summary causal graph as Figure(c) below shows.


<img width="611" alt="Summary causal graph" src="https://github.com/user-attachments/assets/0e4ca3fe-4557-4a64-9e96-0b08765b9818">


As for case study real world dataset, can be download from the link below:

- IPSL data portal: https://aims2.llnl.gov/search/cmip6

- NCEP-NCAR reanalysis 1 data portal: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
  
<img width="919" alt="factor model" src="https://github.com/user-attachments/assets/2c382f0c-b689-4538-a59e-e08325dde1b7">

## Correcrtion

Second part is correction which is trying to used the latent confounder learnt by deconfounding part as an additional feature for precipitation 
correction. As descriped in the paper, we choose the SOTA model iTransformer to perform this step. Implementation is given in https://github.com/thuml/iTransformer.

