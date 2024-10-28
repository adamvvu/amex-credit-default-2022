## American Express - Credit Default Prediction Challenge 2022

This was a simple weekend project for predicting credit default in consumer lending, hosted by American Express on Kaggle in 2022[^1]. The goal is to predict the probability that a consumer defaults on their credit card balance in the future 120 days, given their past balances and a large set of consumer-level data (> 5M+ observations). 

However since the data was heavily obfuscated for privacy reasons, manual feature engineering is difficult. In this project, I explored the usage of persistent homology to automatically extract additional topological features that may describe consumer characteristics. To handle temporal dependencies and non-stationarity in the data, I experimented with a combination of Long Short-term Memory recurrent neural networks and a variant on Attention over the temporal dimension as inspired by the Transformer[^2] architecture. To further boost the predictive performance of the model, I used Snapshot Ensembling[^3] to generate an ensemble of neural networks without additional computational expense.

A particularly challenging aspect of the project is the non-standard evaluation metric given by
$$ M = 0.5(G + D) $$
where $G$ is the normalized Gini coefficient, and $D$ is the percentage of true defaults captured within the top 4\% of predicted probabilities. As this metric is non-differentiable and depends on the relative rankings of all estimated probabilities, simply optimizing for standard cross-entropy loss is sub-optimal. In this project, I experimented with the construction of a novel surrogate loss function that aims to serve as a smooth proxy for the metric and obtained decent results. Further tuning of the neural network architecture would likely lead to even better results.


#### Dependencies
- `python >=3.7`
- `tensorflow >= 2`
- `numpy`
- `pandas`
- `dask`
- `scikit-learn`
- `gtda`
- `matplotlib`
- `seaborn`
- `tqdm`

[^1]: Addison Howard, AritraAmex, Di Xu, Hossein Vashani, inversion, Negin, and Sohier Dane. American Express - Default Prediction. https://kaggle.com/competitions/amex-default-prediction, 2022. Kaggle.

[^2]: Vaswani, Ashish, et al. Attention Is All You Need. arXiv:1706.03762, arXiv, 1 Aug. 2023. arXiv.org, http://arxiv.org/abs/1706.03762.

[^3]: Huang, Gao, et al. Snapshot Ensembles: Train 1, Get M for Free. arXiv:1704.00109, arXiv, 31 Mar. 2017. arXiv.org, http://arxiv.org/abs/1704.00109.