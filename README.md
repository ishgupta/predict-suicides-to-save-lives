# predict suicides to save lives
The National Longitudinal Mortality Study (NLMS) is a national, longitudinal, mortality study sponsored by the parts of the National Institutes of Health, and Center for Disease Control and Prevention and the U.S. Census Bureau for the purpose of studying the effects of differentials in demographic and socio-economic characteristics on mortality.

The main NLMS consists of a database developed for the purpose of studying the effects of demographic and socio-economic characteristics on differentials in U.S. mortality rates. It consists of U.S. Census Bureau data from Current Population Surveys (CPS) and a subset of the 1980 Census combined with death certificate information to identify mortality status and cause of death. The study currently has approximately 3.8 million records with over 550,000 identified mortality cases. The content of the socio-economic variables available offers researchers the potential to answer questions on mortality differentials for a variety of key socio-economic and demographic subgroups not covered as extensively in other databases. 
Mortality information is obtained from death certificates available for deceased persons through the National Center for Health Statistics. Standard demographic and socio-economic variables such as education, income, and employment, as well as information collected from death certificates including cause of death are available for analyses. 

## datasets and inputs
The 11-year follow up consists of a subset of the 39 NLMS cohorts included in the full NLMS that can be followed prospectively for 11 years. The content of each record on the file includes demographic and socioeconomic variables combined with a mortality outcome, if there is one. To prevent disclosure, all of the records have been concatenated into a single file and the temporal dimension has been altered. In lieu of identifying the CPS year and starting point of mortality follow-up for each file, all of the records in have been assigned an imaginary starting point conceptually identified as April 1, 1990. These records are then tracked forward for 11 years to observe whether person in the file has died. This approach results in a maximum of 4018 days of follow up for this cohort.

For those who have died, the underlying cause of death and follow-up time until death have been provided. For those not deceased by the end of 4018 days follow-up period, the follow-up time provided is the full observation length, 4018 days or 11 years. In the construction of data, it was assumed that these surveys, collected from throughout the 1980s and 1990s, would adequately reflect the U.S. non-institutionalized population on April 1, 1990. Under this assumption, the separate CPS samples have been combined and can be viewed as one large sample taken on that date.
The data attributes are explained [here](https://github.com/ishgupta/predict-suicides-to-save-lives/blob/main/docs/Reference%20Manual%20Version%205.docx).

### data
The dataset can be found [here](https://github.com/ishgupta/predict-suicides-to-save-lives/blob/main/data_/data.csv).

### project structure
The project contains 3 main scripts:
0. ml_toolkit.py : a utility script having general ready-to-use functionalities which is utilized by all 3 scripts
1. eda.ipynb : involves EDA
2. process_xg_classifier.ipynb : involves experience with the XGBoost classifier
3. process_nn_classifier.ipynb : involves experience with Pytorch model build for classification

# setup
please make sure you have below packages installed to use the script:
1. xgboost
2. sklearn
3. pandas
4. numpy
5. torch
6. matplotlib
7. jupyter

# dataset
1. 11.csv - used by EDA script could not be included in github repo, or the zip file because of size limitations, and has been uploaded [here](https://drive.google.com/file/d/1d7Aqwn8Vzj5z9c19XAXfWOPkiHcgEn4N/view?usp=sharing) for reference. Please load it into "data_/" directory of the project to start with the eda.ipynb script
2. data_filtered_imputed_clean.csv - exported after pre-processing in the eda.ipynb is used as input by other scripts