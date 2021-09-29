# Predicting COVID-19 Patient Shielding: A Comprehensive Study 

This repository contains code used for:

**Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2021). Predicting COVID-19 Patient Shielding: A Comprehensive Study. The 34th Australasian Joint Conference on Artificial Intelligence (AJCAI 2021). (to appear)**

The Coronavirus disease 2019 (COVID-19) pandemic has presented a considerable challenge to the world health care system, and the management of COVID-19 is an ongoing struggle. The ability to identify and protect high-risk groups is debated by the scientific community. The [United Kingdom NHS Digital](https://digital.nhs.uk/coronavirus/shielded-patient-list/methodology/) published a system to identify patients who meet high-risk criteria of COVID-19 and a framework to approach risk assessments.    

This research uses such publicly published information and predicts medical codes that are listed as criteria for identifying high-risk categories from EHRs. Due to privacy and legal issues, obtaining current patient records from hospitals is not possible. However, EHRs from [MIMIC-III](https://physionet.org/works/MIMICIIIClinicalDatabase/access.shtml) and [eICU](https://eicu-crd.mit.edu/) are used for this research. ICD-9 codes associated with COVID-19 patient shielding for MIMIC-III data with recorded discharge summary is shown in the figure below. The level 1 ICD-9 groups associated with the codes to provide an understanding of the hierarchical grouping of the specific labels are also included. 

![covidtree](https://user-images.githubusercontent.com/60803118/135203626-9684f468-504f-48ec-a924-3e993474f070.png)


