# Kaggle CxR Project

## Goals
<p>Differentiate the X-ray image among Covid-19, Normal, Pneumonia and Tuberculosis</p>

### Data

<p>The data are different in size but with same uint8 dtype, we can just resize and rescale the data</p>

<p>However, from a perspective of a former radiographer, lots of images have really bad image quality.
Some of the data are perhapse screenshots and some of the data have unnecessary digital info on top.</p>
<p>Also, most of the normal subjects and patients may not have the same x-ray routine procedures which I listed in table below, my greaest concern are:</p>
<ol>
   <li> model might recognised the foreign bodies, instead of the image patterns on lungs </li>
   <li> model might recognised the images with poor image-quality as patients directly</li>
</ol>

Infos   | Normal people | Patients
---|---|---
Forein bodies | Less likely to have | Higher chances to have EKG leads, endo, etc. to monitor their vital signs
Image Quality | Most likely in a stand position with bucky (Better image quality) | Most likely in a supine position without bucky (Poor image quality)
