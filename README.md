# Covid-Twitter-prediction
This work was done under the supervision of Dr Saptarshi Ghosh at the Departmnet of Computer Science and Engineering, IIT Kharagpur.<br /><br />
The work aims at using Natural Language Processing techniques to identify the different types of Covid symptom reporting tweets posted from 2020-2021, and use it to predict the number of Covid cases and deaths (in advance) at both Global and Indian level.<br />
<br />
This repository contains the script and files used in the project with the following directories:
<br />
* [files](files) - consisting of the keywords list used to fetch Twitter posts and symptom keywords used to filter tweets with Covid symptom keywords
* [data_sample](data_sample) - consisting of the data samples corresponding to the following four classes of Symptom Reporting Tweet used in the work. A ‘Symptom Reporting Tweet’ is a tweet that reports that some person is experiencing COVID-19 symptoms, regardless of whether the said person has actually been tested for COVID-19. <br />
  * primary reporting - where the writer (person who wrote the tweet) is reporting symptoms of himself/herself
  * secondary reporting - where the writer is reporting symptoms of some friend/relative /neighbor /family member/someone whom the writer is closely related to
  * third-party reporting - where the writer is reporting symptoms of some celebrity/third-party person
  * non-reporting/general reporting - where the writer is not reporting symptoms, but rather talking about symptoms in some other context
* [data_preparation](data_preparation)
  * [classifier](/data_preparation/classifier) - consisting of the scripts used to fetch Twitter posts, filter them with Covid symptom keywords, remove duplicate posts, and prepare training data.<br />
  * [analysis](/data_preparation/analysis) - consists of the script to prepare rolling data for correlation analysis and prediction modeling. <br />
* [classifier](classifier)
  * [single_stage_classifier](/classifier/single_stage_classifier) - consisting of training, validation and inference scripts for classification of tweets into primary, secondary, third-party and non-reporting categories.
  * [two_stage_classifier](/classifier/two_stage_classifier) - consisting of the validation script for two stage classification which involves classifying a tweet as either symptom reporting or non-reporting (stage 1), and if symptom reporting, then classification into primary, secondary and third-party reporting (stage 2).
  * [custom_classifier](/classifier/custom_classifier) - consisting of the single-stage training, validation and inference scripts for multiclass classification of tweets using handcrafted features.
  * [baseline_classifier](/classifier/baseline_classifier) - consisting of script used to train Random Forest and Support Vector Classifier with Tf-Idf embeddings. Also contains script for Support Vector Classifier with Fasttext emneddings.
* [analysis](analysis)
  * [correlations](/analysis/correlations) - consisting of the script used to calculate Pearson and Spearman time-lagged correlations between the cumulative number of different types of symptom reporting tweets posted and the actual number of Covid cases and deaths occuring at a daily level.
  * [predictions_and_plots](/analysis/predictions_and_plots) - consisting of the script used to build Lasso and Polynomial regression models using social media signals to predict Covid cases and deaths in advance.

The following packages were used used in the above work:<br />
* python==3.8
* pytorch==1.6
* torchtext==0.8
* transformers==4.1

For queries or issues feel free to contact me at: [Email ID](mailto:ritikaagarwal19@ap.ism.ac.in)
