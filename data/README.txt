This dataset is extracted from NUS-WIDE.
NUS-WIDE contains 269,648 images from Flickr,
each associated with 6 tags in average. We refer to the image and
its tags as an image-text pair. There are 81 ground truth concepts
manually annotated for evaluation. Following previous works  
we extract 190,421 image-text pairs annotated with the most
frequent 21 concepts and split them into three subsets for training,
validation and test respectively. The size of each subset is shown
in Table 1. For validation (resp. test), 100 (resp. 1000) queries
are randomly selected from the validation (resp. test) dataset. Image
and text features have been provided in the dataset. For
images, SIFT features are extracted and clustered into 500 visual
words. Hence, an image is represented by a 500 dimensional bagof-
visual-words vector. Its associated tags are represented by a
1, 000 dimensional tag occurrence vector.

Table 1:
Dataset: 				NUS-WIDE 
Total size: 			190,421
Training set: 			60,000
Validation set: 		10,000
Test set: 				120,421
Average Text Length: 	6

Phase		File				Shape
Training	trainImg.npy		60000*500
			trainTxt.npy		60000*1000

Validation	validationImg.npy	10000*500
			validationTxt.npy	10000*1000
			validationGnd.npy	10000*21
	
Test		testImg.npy			120421*500
			testTxt.npy			120421*1000
			testGnd.npy			120421*21

Note: 
*Gnd.npy is the label file
image features have already whitened through ZCA, i.e., normalized.
both validation and test query files are generated randomly online