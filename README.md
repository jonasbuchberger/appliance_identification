# Appliance Identification Benchmark

Project contains the preprocessing of the BLUED dataset for event-based Non-Intrusive Load Monitoring testing purposes. In addition two algroithms were recreated and tested.

## Getting Started

Simply clone or download the repository. The package requirements for running the project can be found in the requirements.txt.

### Prerequisites

For preprocessing the data, 20 GB of storage are temporary needed. More than 40 GB of ram would be useful. The preprocessed data has about 3 GB.

### Installing

After cloning or downloading, detlete the sample.txt files.

## Running the tests

The project contains a makefile with the following commands:

* make data: Downloads and preprocesses the BLUED.
* make test-pca: Tests the PCA classifier.
* make test-trees: Tests the Extra Tree classifier.

### Break down into end to end tests

The tests on both algortihm show the disaggregation performance on real household energy measurements. Multiple feature combinations and paratmeter variations are tested.

## Authors

* **Jonas Buchberger** - *Initial work* 

## Acknowledgments

* Extra Tree: https://ieeexplore.ieee.org/document/8284200/
* PCA: https://ieeexplore.ieee.org/document/7997812


