# Kaggle CxR Project

## Goals
<p>Differentiate the X-ray image among Covid-19, Normal, Pneumonia and Tuberculosis</p>

### Data

<p>The data are different in size but with same uint8 dtype, we can just resize and rescale the data</p>

<p>However, from a perspective of a former radiographer, lots of images have really bad image quality.
Some of the data are perhapse screenshots and some of the data have unnecessary digital info on top.
Also, most of the normal subjects and patients may not have the same x-ray routine procedures</p>

   |Normal person | Patient
---|---|---
Forein bodies | Less likely to have | Higher chances to have EKG lead, endo, etc. to monotor their vital signs
Image Quality | Most likely in a stand position with bucky (Better image quality) | Most likely in a supine position without bucky (Poor image quality)