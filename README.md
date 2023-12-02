# Codes for implementation of DH-SNNs  

  This code implements the DH-SNNs for various tasks. We select some typical training codes for tasks in the paper to present. 

1) files and folders description: 
   
* The pre-processing and training codes can be found in the folder that corresponds to the task. (The folder named "delayed_xor" and "multitimescale_xor" represent the self-designed delayed spiking XOR problem and multi-timescale spiking XOR problem, respectively.)

* The folder named "SNN_layers" contains the main codes for the implementation of DH-SNNs model.


2) The datasets: 
	1. SHD and SSC datasets can be downloaded from https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
	2. GSC can be downloaded from https://tensorflow.google.cn/datasets/catalog/speech_commands/
  	3. (P)S-MNIST: This dataset can be found in torchvision.datasets.MNIST
	4. DEAP can be downloaded from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ 
	5. TIMIT can be found here: https://catalog.ldc.upenn.edu/LDC93S1
	6. Self-designed delayed spiking XOR problem and multi-timescale spiking XOR problem.
	7. NeuroVPR task

3) Pre-requisites 

- Python >=3.9
- HDF5 for python.
- Pytorch == 1.12.1, torchvision == 0.13.1, torchaudio == 0.12.1
- Preprocessing packages: [librosa]([librosa — librosa 0.8.0 documentation](https://librosa.org/doc/latest/index.html)),[tables]([tables · PyPI](https://pypi.org/project/tables/)),[wfdb]([wfdb — wfdb 3.3.0 documentation](https://wfdb.readthedocs.io/en/latest/)),[klepto]([klepto · PyPI](https://pypi.org/project/klepto/)) and [Scipy]([SciPy.org — SciPy.org](https://www.scipy.org/index.html)),
- matplotlib
- scikit-learn == 1.1.1 
- pandas == 1.4.3 
- pickle

4) Code running
* Data preprocessing. 
  
  The datasets(SHD,SSC,GSC,TIMIT and DEAP) are required to arrange the data before training. The pre-processing codes and instructions can be found in the folder that corresponds to the task. 
  The data of NeuroVPR is available on Zenodo: https://zenodo.org/records/7827108#.ZD_ke3bP0ds
  
* Model training. The training codes can be found in the folder that corresponds to the task.
  To start the training of DH-SNNs on SSC, for example, just go to the folder SSC and run
  ```
  # DH-SFNN on SSC
  python  main_dense_denri.py 
  ```
  or
  ```
  # DH-SRNN on SSC
  python  main_rnn_denri.py 
  ``` 

* Pre-trained models are provided for some tasks in the folder. 




