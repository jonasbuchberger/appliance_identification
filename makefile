.PHONY: data test-pca test-trees
   
data:
	python src/event_extraction.py

test-pca:
	python -W ignore src/test_pca.py

test-trees:
	python -W ignore src/test_extra_trees.py