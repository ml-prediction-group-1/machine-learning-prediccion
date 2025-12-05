train:
	python3 src/training.py

predict:
	python3 src/prediction.py

clean:
	rm -rf __pycache__
