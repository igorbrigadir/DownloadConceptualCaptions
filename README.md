# Download Conceptual Captions Data

Place data from: https://ai.google.com/research/ConceptualCaptions/download in this folder

`Train_GCC-training.tsv` Training Split (3,318,333)

`Validation_GCC-1.1.0-Validation.tsv` Validation Split (15,840)

Test Split (~12,500) human approved image caption pairs is not public.

run `download_data.py`

Images will be in `training` and `validation` folders. You can stop and resume, the settings for splitting downloads into chunks / threads are not optimal, but it maxed out my connection so i kept them as is.

A bunch of them will fail to download, and return web pages instead. These will need to be cleaned up later. See `downloaded_validation_report.tsv` after it downloads for HTTP errors. Around 8% of images are gone, based on validation set results.

It should take about a day to download the training data, but it's still running so i can't say for sure.
