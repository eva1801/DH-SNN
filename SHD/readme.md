use shd_generate_dataset.py to preprocess raw data of SHD first

main_dense_orgin.py for vanilla SFNNs on the SHD and main_rnn_origin.py for vanilla SRNNs
main_dense_denri.py for DH-SFNNs on the SHD and main_rnn_denri.py for DH-SRNNs 

To start the training of DH-SNNs, for example, just run the following commands
  ```
  # DH-SFNN 
  python  main_dense_denri.py 
  ```
  or
  ```
  # DH-SRNN
  python  main_rnn_denri.py 
  ``` 
  or
  ```
  # vanilla-SFNN 
  python  main_dense_origin.py 
  ```
  ```
  # vanilla-SRNN 
  python  main_rnn_origin.py 
  ```