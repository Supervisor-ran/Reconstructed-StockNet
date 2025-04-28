# Reconstructed-StockNet
StockNet presents a clear pipeline for processing both time-series data, such as stock prices, and text data collected from X (formerly Twitter). In addition to utilizing an attention mechanism within word embeddings, it also integrates an attention mechanism into the loss function, which accounts for latent states.
However, despite being published in 2018, StockNet was coded in the very outdated Python 2 and TensorFlow 1.4.0 (https://github.com/yumoxu/stocknet-code). Therefore, while studying the architecture of StockNet, I rebuilt it using Python 3.10 and PyTorch 2.3.1.
The code architecture was restructured as follows: Compared to the old version, the three-layer model is not integrated into a single module but is instead constructed in a modular and separated manner. This approach enhances maintainability.

StrockNet (https://github.com/yumoxu/stocknet-dataset) dataset could be directly used in this code. 

Original paper is here. https://aclanthology.org/P18-1183/
