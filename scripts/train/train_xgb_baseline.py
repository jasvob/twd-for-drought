import sys
sys.path.append('../')

import numpy as np

from omegaconf import OmegaConf

from utils.data.treenet_dataframe import TreeNetDataFrame
from ml.data.treenet_dataset import TreeNetTemporalDataset

from sklearn.ensemble import GradientBoostingRegressor

import pickle


import argparse
from pathlib import Path

# Prepare command line argument parser
arg_parser = argparse.ArgumentParser(
    prog='Train Gradient Boosting Regressor Baseline',
    description='Train GBR baseline and save model to file.',
)

arg_parser.add_argument('--config', type=str, help='Path to the configuration folder.')   
arg_parser.add_argument('--output_path', type=str, help='Path to where the outputs should be stored.')

if __name__ == "__main__":
    args = arg_parser.parse_args()

    cfg = OmegaConf.load(args.config)

    cfg_model = cfg['model']
    cfg_train_dataset = cfg['data']['init_args']['train_dataset']['init_args']
    cfg_test_dataset = cfg['data']['init_args']['test_dataset']['init_args']

    dataset_train = TreeNetTemporalDataset(
        treenet_data_frame=TreeNetDataFrame(
            data_file_path=cfg_train_dataset['treenet_data_frame']['init_args']['data_file_path'],
            metadata_file_path=cfg_train_dataset['treenet_data_frame']['init_args']['metadata_file_path'],
            metadata_filter_fn=cfg_train_dataset['treenet_data_frame']['init_args']['metadata_filter_fn'],
            treedata_filter_fn=cfg_train_dataset['treenet_data_frame']['init_args']['treedata_filter_fn'],
            cache_path=cfg_train_dataset['treenet_data_frame']['init_args']['cache_path'],
            data_filter_fn=cfg_train_dataset['treenet_data_frame']['init_args']['data_filter_fn'],
            min_trees_per_site=cfg_train_dataset['treenet_data_frame']['init_args']['min_trees_per_site'],
            equal_sequence_per_site=cfg_train_dataset['treenet_data_frame']['init_args']['equal_sequence_per_site'],
            interpolate_data=cfg_train_dataset['treenet_data_frame']['init_args']['interpolate_data'],
            drop_na_for_variables=cfg_train_dataset['treenet_data_frame']['init_args']['drop_na_for_variables'],
            drop_na_for_metadata=cfg_train_dataset['treenet_data_frame']['init_args']['drop_na_for_metadata'],
            resample_data=cfg_train_dataset['treenet_data_frame']['init_args']['resample_data'],
            filter_growth_period=cfg_train_dataset['treenet_data_frame']['init_args']['filter_growth_period'],
            filter_by_temperature=cfg_train_dataset['treenet_data_frame']['init_args']['filter_by_temperature'],
            filter_by_frost=cfg_train_dataset['treenet_data_frame']['init_args']['filter_by_frost'],
            filter_min_growth=cfg_train_dataset['treenet_data_frame']['init_args']['filter_min_growth'],
            filter_min_twd=cfg_train_dataset['treenet_data_frame']['init_args']['filter_min_twd'],
            merge_by_site=cfg_train_dataset['treenet_data_frame']['init_args']['merge_by_site'],
        ),
        input_feats=cfg_train_dataset['input_feats'],
        exog_feats=cfg_train_dataset['exog_feats'],
        target_feats=cfg_train_dataset['target_feats'],
        seq_len=cfg_train_dataset['seq_len'],
        max_dist_timesteps=cfg_train_dataset['max_dist_timesteps'],
        normalization=cfg_train_dataset['normalization'],
        norm_inputs=cfg_train_dataset['norm_inputs'],
        norm_exog=cfg_train_dataset['norm_exog'],
        norm_targets=cfg_train_dataset['norm_targets'],
        target_only_growth=cfg_test_dataset['target_only_growth'],
        return_timestamp=True
    )

    dataset_test = TreeNetTemporalDataset(
        treenet_data_frame=TreeNetDataFrame(
            data_file_path=cfg_test_dataset['treenet_data_frame']['init_args']['data_file_path'],
            metadata_file_path=cfg_test_dataset['treenet_data_frame']['init_args']['metadata_file_path'],
            metadata_filter_fn=cfg_test_dataset['treenet_data_frame']['init_args']['metadata_filter_fn'],
            treedata_filter_fn=cfg_test_dataset['treenet_data_frame']['init_args']['treedata_filter_fn'],
            cache_path=cfg_test_dataset['treenet_data_frame']['init_args']['cache_path'],
            data_filter_fn=cfg_test_dataset['treenet_data_frame']['init_args']['data_filter_fn'],
            min_trees_per_site=cfg_test_dataset['treenet_data_frame']['init_args']['min_trees_per_site'],
            equal_sequence_per_site=cfg_test_dataset['treenet_data_frame']['init_args']['equal_sequence_per_site'],
            interpolate_data=cfg_test_dataset['treenet_data_frame']['init_args']['interpolate_data'],
            drop_na_for_variables=cfg_test_dataset['treenet_data_frame']['init_args']['drop_na_for_variables'],
            drop_na_for_metadata=cfg_test_dataset['treenet_data_frame']['init_args']['drop_na_for_metadata'],
            resample_data=cfg_test_dataset['treenet_data_frame']['init_args']['resample_data'],
            filter_growth_period=cfg_test_dataset['treenet_data_frame']['init_args']['filter_growth_period'],
            filter_by_temperature=cfg_test_dataset['treenet_data_frame']['init_args']['filter_by_temperature'],
            filter_by_frost=cfg_test_dataset['treenet_data_frame']['init_args']['filter_by_frost'],
            filter_min_growth=cfg_test_dataset['treenet_data_frame']['init_args']['filter_min_growth'],
            filter_min_twd=cfg_test_dataset['treenet_data_frame']['init_args']['filter_min_twd'],
            merge_by_site=cfg_test_dataset['treenet_data_frame']['init_args']['merge_by_site'],
        ),
        input_feats=cfg_test_dataset['input_feats'],
        exog_feats=cfg_test_dataset['exog_feats'],
        target_feats=cfg_test_dataset['target_feats'],
        seq_len=cfg_test_dataset['seq_len'],
        max_dist_timesteps=cfg_test_dataset['max_dist_timesteps'],
        normalization=cfg_test_dataset['normalization'],
        norm_inputs=cfg_test_dataset['norm_inputs'],
        norm_exog=cfg_test_dataset['norm_exog'],
        norm_targets=cfg_test_dataset['norm_targets'],
        target_only_growth=cfg_test_dataset['target_only_growth'],
        return_timestamp=True
    )

    all_inps = []
    all_exog = []
    all_tgts = []
    for i in range(len(dataset_train)):
        inps, exog, tgts, _, _, _ = dataset_train[i]
        all_inps.append(inps.numpy())
        all_exog.append(exog.numpy())
        all_tgts.append(tgts.numpy())
    all_inps = np.stack(all_inps)
    all_exog = np.stack(all_exog)
    all_tgts = np.stack(all_tgts)


    all_inps_test = []
    all_exog_test = []
    all_tgts_test = []
    for i in range(len(dataset_test)):
        inps, exog, tgts, _, _, _ = dataset_test[i]
        all_inps_test.append(inps.numpy())
        all_exog_test.append(exog.numpy())
        all_tgts_test.append(tgts.numpy())
    all_inps_test = np.stack(all_inps_test)
    all_exog_test = np.stack(all_exog_test)
    all_tgts_test = np.stack(all_tgts_test)

    backcast_len = cfg_model['backcast_len']
    forecast_len = cfg_model['forecast_len']

    rf = GradientBoostingRegressor(loss = cfg_model['loss'], alpha = cfg_model['alpha'], max_depth = cfg_model['max_depth'], min_samples_leaf = cfg_model['min_samples_leaf'],
                        min_samples_split = cfg_model['min_samples_split'], n_estimators = cfg_model['n_estimators'], random_state = 42, verbose=100)

    print('Fitting XGB ...')        
    rf_inputs = np.concatenate([all_inps[:, :backcast_len, :].reshape(all_inps.shape[0], -1), all_tgts[:, :backcast_len, :].reshape(all_tgts.shape[0], -1), all_inps[:, -forecast_len:, :].reshape(all_inps.shape[0], -1)], axis=-1)
    rf_targets = all_tgts[:, -forecast_len:, :].reshape(all_tgts.shape[0], -1)

    model = rf.fit(rf_inputs, rf_targets)
    print('Done.')

    print('Saving XGB ...')
    output_path = Path(args.output_path).joinpath(args.config.stem)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = 'model.pkl' 
    pickle.dump(model, open(output_path.joinpath(filename), 'wb'))
    print('Done.')

    print('Evaluating XBG ...')        
    rf_inputs_test = np.concatenate([all_inps_test[:, :backcast_len, :].reshape(all_inps_test.shape[0], -1), all_tgts_test[:, :backcast_len, :].reshape(all_tgts_test.shape[0], -1), all_inps_test[:, -forecast_len:, :].reshape(all_inps_test.shape[0], -1)], axis=-1)
    rf_targets_test = all_tgts_test[:, -forecast_len:, :].reshape(all_tgts_test.shape[0], -1)

    preds = model.predict(rf_inputs_test)
    print('MSE: ', np.sqrt(((preds - rf_targets_test)**2).sum(axis=-1)).mean())
    print('Done.')
