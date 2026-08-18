
import torch
import numpy as np

from typing import List

def load_data(data_path : str, experiment_name : str):
    print('Loading \'%s\'' % experiment_name)
    dat = np.load('%s/%s.npz' % (data_path, experiment_name), allow_pickle=True)['results'].item()

    inps = torch.from_numpy(dat['inps'])
    exogs = torch.from_numpy(dat['exogs'])
    bcast_tgts = torch.from_numpy(dat['bcast_tgts'])
    fcast_tgts = torch.from_numpy(dat['fcast_tgts'])
    forecasts = torch.from_numpy(dat['forecasts'])
    if 'decomps' in dat.keys():
        decomps = torch.from_numpy(dat['decomps'])
    else:
        decomps = None
    if 'residuals' in dat.keys():
        residuals = torch.from_numpy(dat['residuals'])
    else:
        residuals = None
    if 'decomps_bcast' in dat.keys():
        decomps_bcast = torch.from_numpy(dat['decomps_bcast'])
    else:
        decomps_bcast = None
    tstamps = torch.from_numpy(dat['tstamps'])
    site_ids = torch.from_numpy(dat['site_ids'])
    series_ids = torch.from_numpy(dat['series_ids'])

    data_dict = {
        'inps': inps,
        'exogs': exogs, 
        'bcast_tgts': bcast_tgts,
        'fcast_tgts': fcast_tgts,
        'forecasts': forecasts,
        'decomps': decomps,
        'residuals': residuals,
        'decomps_bcast': decomps_bcast,
        'tstamps': tstamps,
        'site_ids': site_ids,
        'series_ids': series_ids
    }

    return data_dict

def load_attributions(data_path : str, experiment_name : str, series_list : List[str], only_attribution : bool = True):
    all_attributions = []
    all_vars = []
    backcast_len = []
    forecast_len = []
    input_feats = []
    target_feats = []
    all_preds = []
    all_tstamps = []
    for series_id in series_list:
        dat = np.load('%s/%s_%s_all_grshap.npz' % (data_path, experiment_name, series_id), allow_pickle=True)['results'].item()

        all_attributions.append(torch.from_numpy(dat['all_attributions']))
        backcast_len.append(dat['backcast_len'])
        forecast_len.append(dat['forecast_len'])
        input_feats.append(dat['input_feats'])
        target_feats.append(dat['target_feats'])
        if not only_attribution:
            all_vars.append(torch.from_numpy(dat['all_vars']))
            all_preds.append(torch.from_numpy(dat['pred']))
            all_tstamps.append(torch.from_numpy(dat['tstamp']))
        
    all_attributions = torch.cat(all_attributions, dim=0)
    backcast_len = backcast_len[0]
    forecast_len = forecast_len[0]
    input_feats = input_feats[0]
    target_feats = target_feats[0]

    if not only_attribution:
        all_vars = torch.cat(all_vars, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_tstamps = torch.cat(all_tstamps, dim=0)


    attr_dict = {
        'attributions': all_attributions,
        'backcast_len': backcast_len,
        'forecast_len': forecast_len,
        'input_feats': input_feats,
        'target_feats': target_feats
    }

    if not only_attribution:
        attr_dict['vars'] = all_vars
        attr_dict['preds'] = all_preds
        attr_dict['tstamps'] = all_tstamps

    return attr_dict
    

def load_attributions_per_series(data_path : str, experiment_name : str, series_list : List[str], only_attribution : bool = True):
    all_attributions = []
    all_vars = []
    backcast_len = []
    forecast_len = []
    input_feats = []
    target_feats = []
    all_preds = []
    all_tstamps = []
    for series_id in series_list:
        dat = np.load('%s/%s_%s_all_grshap.npz' % (data_path, experiment_name, series_id), allow_pickle=True)['results'].item()

        all_attributions.append(torch.from_numpy(dat['all_attributions']))
        backcast_len.append(dat['backcast_len'])
        forecast_len.append(dat['forecast_len'])
        input_feats.append(dat['input_feats'])
        target_feats.append(dat['target_feats'])
        if not only_attribution:
            all_vars.append(torch.from_numpy(dat['all_vars']))
            all_preds.append(torch.from_numpy(dat['pred']))
            all_tstamps.append(torch.from_numpy(dat['tstamp']))
        
    all_attributions = all_attributions
    backcast_len = backcast_len[0]
    forecast_len = forecast_len[0]
    input_feats = input_feats[0]
    target_feats = target_feats[0]

    if not only_attribution:
        all_vars = all_vars
        all_preds = all_preds
        all_tstamps = all_tstamps


    attr_dict = {
        'attributions': all_attributions,
        'backcast_len': backcast_len,
        'forecast_len': forecast_len,
        'input_feats': input_feats,
        'target_feats': target_feats
    }

    if not only_attribution:
        attr_dict['vars'] = all_vars
        attr_dict['preds'] = all_preds
        attr_dict['tstamps'] = all_tstamps

    return attr_dict
    