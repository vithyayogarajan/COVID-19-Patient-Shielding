# Predicting COVID-19 Patient Shielding: A Comprehensive Study 

This repository contains code used for:

**Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2021). Predicting COVID-19 Patient Shielding: A Comprehensive Study. The 34th Australasian Joint Conference on Artificial Intelligence (AJCAI 2021). (to appear)**

The Coronavirus disease 2019 (COVID-19) pandemic has presented a considerable challenge to the world health care system, and the management of COVID-19 is an ongoing struggle. The ability to identify and protect high-risk groups is debated by the scientific community. The [United Kingdom NHS Digital](https://digital.nhs.uk/coronavirus/shielded-patient-list/methodology/) published a system to identify patients who meet high-risk criteria of COVID-19 and a framework to approach risk assessments.    

This research uses such publicly published information and predicts medical codes that are listed as criteria for identifying high-risk categories from EHRs. Due to privacy and legal issues, obtaining current patient records from hospitals is not possible. However, EHRs from [MIMIC-III](https://physionet.org/works/MIMICIIIClinicalDatabase/access.shtml) and [eICU](https://eicu-crd.mit.edu/) are used for this research. ICD-9 codes associated with COVID-19 patient shielding for MIMIC-III data with recorded discharge summary is shown in the figure below. The level 1 ICD-9 groups associated with the codes to provide an understanding of the hierarchical grouping of the specific labels are also included. 

<img src="https://user-images.githubusercontent.com/60803118/135203626-9684f468-504f-48ec-a924-3e993474f070.png" alt="covidtree" width="750"/>

## Classifiers and Language Models

**'Traditional' classifiers** such as Binary Relevance (BR) and Classifier Chains (CC) use [MEKA](http://waikato.github.io/meka/). Example scripts are provided above. 

 **Pre-trained word embeddings based Neural Networks** 
 1. [CAML and DRCAML](https://github.com/jamesmullenbach/caml-mimic)  
 2. [CNNText](https://arxiv.org/abs/1408.5882) example code provided. 
 
**Transformers**
Models used include:[BERT-base](https://github.com/google-research/bert),[Clinical BERT](https://github.com/EmilyAlsentzer/clinicalBERT),[BioMed-RoBERTa](https://huggingface.co/allenai/biomed_roberta_base),[PubMedBERT](https://microsoft.github.io/BLURB/models.html),[MeDAL-Electra](https://github.com/BruceWen120/medal),[Longformer](https://github.com/allenai/longformer) and [TransformerXL](https://github.com/kimiyoung/transformer-xl)
