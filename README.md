# Covid-19_Chest_X-Ray

# Background of this work
COVID is possibly better diagnosed using radiological imaging Fang, 2020. Companies are developing AI tools and deploying them at hospitals Wired 2020. We should have an open database to develop free tools that will also provide assistance.

# About the dataset
This dataset is a database of COVID-19 cases with chest X-ray or CT images. It contains COVID-19 cases as well as MERS, SARS, and ARDS.Source of this dataset is kaggle (https://www.kaggle.com/bachrr/covid-chest-xray). Here is a list of each metadata field, with explanations:

* Patientid (internal identifier, just for this dataset)
offset (number of days since the start of symptoms or hospitalization for each image, this is very important to have when there are multiple images for the same patient to track progression while being imaged. If a report says "after a few days" let's assume 5 days.)
sex (M, F, or blank)
age (age of the patient in years)
finding (which pneumonia)
survival (did they survive? Y or N)
view (for example, PA, AP, or L for X-rays and Axial or Coronal for CT scans)
modality (CT, X-ray, or something else)
date (date the image was acquired)
location (hospital name, city, state, country) importance from right to left.
filename
doi (DOI of the research article
url (URL of the paper or website where the image came from)
license
clinical notes (about the radiograph in particular, not just the patient)
other notes (e.g. credit)
# Target of this task
By the use these images in Kaggle kernels to develop AI-based approaches to predict and understand COVID-19.

Let's START!!!!!!
