# Forecasting Tree Water Deficit and Forest Drought Using Explainable Deep Learning
Repository for the paper Forecasting Tree Water Deficit and Forest Drought Using Explainable Deep Learning.

This repository allows to reproduce results from the paper as well as run new experiments using the methodologies provided in the paper.

## Necessary data

This repository works with data which are stored on Zenodo under the following link:

[https://doi.org/10.5281/zenodo.22006871](https://doi.org/10.5281/zenodo.22006871)

Both `treenet-dataset.zip` and `resources.zip` should be downloaded. 

The data should be unpacked into the following folder structure within the repository:
```
resources/
resources/treenet-dataset/
```

## 1. Setting up the environment
To setup the environment necessary to run the code within this repository, please use the `environment.yml` file. You can setup e.g. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment as follows:
```
conda env create -f environment.yaml
```

This will create an environment with the name `twd-env`, which you can then activate by running
```
conda activate twd-env
```

## 2. Training models
Each model has its configuration file, which is available within the `configs/` folder. You can use those configuration files as an inspiration, shall you desire to use this repository to train a different model or change model configuration.

### Deep learning models
Model configuration and training is facilitated through PyTorch [Lightning CLI](https://lightning.ai/). Training run can be started via the Lightning CLI as follows:
```
python main.py fit -c configs/nhits.yaml 
```

### Baselines
There are two special configuration files, `rf.yaml` and `xgb.yaml`, which are not supposed to work directly with Lightning CLI, but are use within separate scripts, which train baseline models without the use of PyTorch Lightning.

To train the baselines, use the followin scripts:
```
PYTHONPATH=. scripts/train/train_rf_baseline.py --config configs/rf.yaml --output_path experiments/
PYTHONPATH=. scripts/train/train_xgb_baseline.py --config configs/xgb.yaml --output_path experiments/
```

## 3. Evaluation
Trained models can be evaluated using scripts within `scripts/eval/` folder. An example run is below:
```
PYTHONPATH=. python scripts/eval/eval_nhits.py --config configs/nhits.yaml --model_ckpt resources/models/nhits.ckpt --output_path resources/
```

## 4. Explainability
To explain predictions of our models, we use Gradient SHAP method provided within [Captum](https://captum.ai/). To run model explainability evaluation, you can use the scripts within `scripts/explain/`. An example run is below. Notice that you have to choose the series for which you want to explain the model predictions.
```
PYTHONPATH=. python scripts/explain/explain_nhits.py --config configs/nhits.yaml --model_ckpt resources/models/nhits.ckpt --output_path resources/ --series_id 1
```

## 5. Visualizations
To produce the figures and tables within the paper, several scripts are provided within `scripts/viz/`.

Performance tables and plot of performance of various tree species over forecasting horizon can be plotted using:
```
PYTHONPATH=. python scripts/viz/plot_performance.py --data_path resources/dataset-treenet/ --output_path resources/ --model nhits
```

Qualitative visualizations of some forecasts can be plotted using:
```
PYTHONPATH=. python scripts/viz/plot_forecasts.py --data_path resources/dataset-treenet/ --output_path resources/ --model nhits
```

Explanations of predictions of the model for one series over time can be plotted with:
```
PYTHONPATH=. python scripts/viz/plot_series_explanation.py --data_path resources/dataset-treenet/ --output_path resources/ --series_id 160 --model nhits
```

Analysis of influence of different variables on TWD can be generated using:
```
PYTHONPATH=. python scripts/viz/plot_variable_explanation.py --data_path resources/dataset-treenet/ --output_path resources/  --model nhits
```
**Disclaimer: Please note that the latter only produces a reduced plot using sample data within `resources/explain/`, as storing the full explanations for all series within this study would require too much space. In order to generate the full figure from the manuscript, you first need to generate tree series explanations using the scripts from above and then make minor changes to the script in order for it to include all tree series.**

## Acknowledgements
This work has been conducted with the support of fundings from:
* Swiss Data Science Center 
* Swiss Federal Institute for Forest, Snow and Landscape Research WSL

## License
Code in this repository is available for research under the CC BY-NC license:
[https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)
