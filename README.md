<image src="logo_A.M.E.L.I.A.png" width="200" align="left"/> 

# A.M.E.L.I.A: Searching for Type II Quasars
Machine Learning pipeline to distinguish SDSS galaxies (includes Seyferts and LINER) and Type II Quasars (QSO2) using supervised and semi-supervised classification.

The logo was created using Dall-E.
<br>
<br>


## How to use
This work was made using Python scripts. 
Each classification scripts depends on two different scripts.
In pipeline_data_preprocessing.py, you will find the functions used to do the necessary pre-processing steps (e.g creating colours).
In pipeline_classification_functions.py, you will find the classifications functions designed for supervised, semi-supervised and unsupervised tasks. Please, used them according to the data set you are using.

### Data

To create the data, we used SQL queries. The data can be retrieved from https://skyserver.sdss.org/CasJobs/ .
You can find the data sets used in this work in the folder data.

### QSO2 classification

The two classification tasks were performed and are described in two Python scripts:
<ul>
  <li> POC supervised: see file "Clf_supervised_POC.py". 
    <li> Semi-supervised: see file "Clf_semisupervised_thr80.py".
</ul>


## Cite us
Thank you for your interest in the A.M.E.L.I.A pipeline.
If this work was helpful, please do not forget to cite us in your publications.
