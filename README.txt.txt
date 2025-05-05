I could not include the cached files because there are too many to upload to GitHub. I might be able to send via google drive link. 
I included a single example of the old training data format => png, jpg and mat file
	- these files start with sun_
There are a couple test files in there as well, such as tensor_info.py and examine_model.py which displays useful information
There are also some .sh scripts that I used on ai-panther
I included the 150th epoch model in the root directory, but also included the others in a sepratate, other_models directory
Inside torch_transformer, I modified only the Affine and Projective Transformers. This is because I was 1, interested in Affine Transformations, but 2 the Tensor Flow model was trained with Projective Transformations, so I rewrote that to PyTorch to get as close to the original implementation as possible. Since I didn't modify the others, I still needed to include an import to tensor flow to not get an error, or I could have removed the other transformers entirely. 
