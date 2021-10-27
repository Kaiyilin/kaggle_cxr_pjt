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

<p>Summary Of the Data</p>
<ul>
   <li> The image data contain numerous scenario in real world, it's realistic but messy<li>
   <li> Might need to convert to gray scale at the first place (better option)</li>
   <li> Imbalance data for traing, the covid-19 data are extremely small:<li>
   <ul>
      <li>NORMAL: 1341</li>
      <li>TURBERCULOSIS: 650</li>
      <li>PNEUMONIA: 3875</li>
      <li>COVID19: 460</li>
   </ul>
   <li>Check the distribution ?</li>
</ul>

### Model and training 
<p>About model</p>
<ul>
   <li> Preliminary with ResNet-50 with input_shape = (256, 256, 1)<li>
   <li> Using SSL(self-supervided learning) might be a better option </li>
   <li> Evaluate and Test with val and test dataset</li>
   <ul>
      <li>val_NORMAL: 8</li>
      <li>val_TURBERCULOSIS: 12</li>
      <li>val_PNEUMONIA: 8</li>
      <li>val_COVID19: 10</li>
      <br>
      <li>test_NORMAL: 234</li>
      <li>test_TURBERCULOSIS: 41</li>
      <li>test_PNEUMONIA: 390</li>
      <li>test_COVID19: 106</li>
   </ul>
</ul>