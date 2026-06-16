.PHONY: test train

test:
	python3 -m unittest discover -s tests

train:
	python3 scripts/train_model.py
