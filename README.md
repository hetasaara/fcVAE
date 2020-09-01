# fcmVI

Flow cytometry has been used for several decades to quantitatively analyse single cells in a high-throughput manner. State-of-the-art flow cytometers allow the detection of more than 20 cellular parameters but the instruments used in clinical practise usually have much more limited capabilities. Thus, flow cytometry samples are often split into separate tubes with varying marker combinations to increase the number of measurable markers. However, this poses challenges to the analysis of flow cytometry data because the data from multiple tubes must be integrated while preserving the original biological information. Currently, most of the computational analysis techniques are not able to handle this kind of multi-tube flow cytometry data. 

Here, we develop a deep generative modelling framework to enable simultaneous integration, clustering, and visualization of such data. We show that the model, named fcmVI, successfully discovers a latent representation of the cell types from flow cytometry data. Furthermore, the fcmVI model can be used to align multiple tubes originating from the same sample in the latent space. The model is applied to two different data sets from mouse immune cells and human acute myeloid leukemia (AML) samples. In addition, the model enables the imputation of missing marker values for each tube, which is demonstrated on both data sets and the results are compared to nearest neighbor imputation. The final model architecture is implemented in Main_fcvae2.ipynb.
