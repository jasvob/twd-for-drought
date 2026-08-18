
import torch
import numpy as np
import sys

from typing import Optional, List, Tuple

def overall_performance(data_dict, filter_sids : Optional[List[int]] = None, experiment_name : str = '', file=sys.stdout):
    if filter_sids is not None:
        mask = np.isin(data_dict['series_ids'], filter_sids)
    else:
        mask = np.ones((data_dict['series_ids'].shape[0], ), dtype=np.bool_)

    quant_idx = data_dict['forecasts'].shape[-1] // 2
    forecast_horizon = data_dict['forecasts'].shape[-2]

    forecasts_filt = data_dict['forecasts'][mask, :forecast_horizon, quant_idx]
    targets_filt = data_dict['fcast_tgts'][mask, :forecast_horizon, 0]
    naive_filt = data_dict['bcast_tgts'][mask, -1:, 0]
    forecasts_filt_clamp = torch.clamp(forecasts_filt, min=1e-6)
    targets_filt_clamp = torch.clamp(targets_filt, min=1e-6)
    naive_filt_clamp = torch.clamp(naive_filt, min=1e-6)

    medianae = torch.median(torch.abs(forecasts_filt_clamp - targets_filt_clamp))
    mae = torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp))
    nmae = torch.sum(torch.abs(forecasts_filt_clamp - targets_filt_clamp)) / torch.sum(torch.abs(targets_filt_clamp)) 
    smape = 100 * torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp) / (torch.abs(forecasts_filt_clamp) + torch.abs(targets_filt_clamp)))
    wsmape = 100 * torch.mean(torch.sum(torch.abs(forecasts_filt_clamp - targets_filt_clamp), dim=1) / torch.sum(torch.abs(forecasts_filt_clamp) + torch.abs(targets_filt_clamp), dim=1))
    male = torch.mean(torch.abs(torch.log(forecasts_filt_clamp / targets_filt_clamp)))
    mase = torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp) / (torch.mean(torch.abs(naive_filt_clamp - targets_filt_clamp), dim=-1, keepdims=True) + 1)) 

    print('All data - %s' % experiment_name, file=file)
    print('-------------------------------------------------------------------------------------------------------', file=file)
    print('MAE \t\t MedianAE \t SMAPE \t\t MASE \t\t NMAE \t\t WMAPE \t\t MALE', file=file)
    print('-------------------------------------------------------------------------------------------------------', file=file)
    print('%.4f \t\t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (mae, medianae, smape, mase, nmae, wsmape, male), file=file)

    medianae_all = torch.median(torch.abs(forecasts_filt_clamp - targets_filt_clamp), dim=0)[0]
    mae_all = torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp), dim=0)
    nmae_all = torch.sum(torch.abs(forecasts_filt_clamp - targets_filt_clamp), dim=0) / (torch.sum(torch.abs(targets_filt_clamp), dim=0) + 1)
    smape_all = 100 * torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp) / (torch.abs(forecasts_filt_clamp) + torch.abs(targets_filt_clamp)), dim=0)
    wsmape_all = 100 * torch.mean(torch.sum(torch.abs(forecasts_filt_clamp - targets_filt_clamp), dim=1) / torch.sum(torch.abs(forecasts_filt_clamp) + torch.abs(targets_filt_clamp), dim=1), dim=0)
    male_all = torch.mean(torch.abs(torch.log(forecasts_filt_clamp / targets_filt_clamp)), dim=0)
    mase_all = torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp) / (torch.mean(torch.abs(naive_filt_clamp - targets_filt_clamp), dim=-1, keepdims=True) + 1), dim=0) 

    return {
        'medianae': medianae_all.numpy(),
        'mae': mae_all.numpy(),
        'nmae': nmae_all.numpy(),
        'smape': smape_all.numpy(),
        'wsmape': wsmape_all.numpy(),
        'male': male_all.numpy(),
        'mase': mase_all.numpy(),
    }

def species_stats(data_dict, metadata_frame, filter_sids : List[int], experiment_name : str = '', file=sys.stdout):  
    print('Per-species statistics - %s' % experiment_name, file=file)
    print('---------------------------------------------------------------------------------------------------------------------------', file=file)
    print('Species Name \t\t\t Support \t Series IDs', file=file)
    print('---------------------------------------------------------------------------------------------------------------------------', file=file)

    meatadata_frame_cat = metadata_frame.copy()
    meatadata_frame_cat['genus_species'] = meatadata_frame_cat.tree_genus + ' ' + meatadata_frame_cat.tree_species

    for key, subdf in meatadata_frame_cat.groupby('genus_species'):
        species_data_points = 0
        species_sids = []
        for sid in subdf.index:
            if sid in filter_sids:
                mask = (data_dict['series_ids'] == sid)
                series_length = mask.sum()
                species_data_points += series_length
                species_sids.append(sid)


        if species_data_points > 0:
            print('%s \t %i \t\t %s' % (key.ljust(20), species_data_points, ', '.join(list(map(str, species_sids)))), file=file)  


def per_species_performance(data_dict, metadata_frame, filter_sids : Optional[List[int]] = None, experiment_name : str = '', file=sys.stdout):  
    quant_idx = data_dict['forecasts'].shape[-1] // 2
    forecast_horizon = data_dict['forecasts'].shape[-2]
    
    print('Per-species data - %s' % experiment_name, file=file)
    print('-------------------------------------------------------------------------------------------------------------------------------', file=file)
    print('Species Name \t\t\t MAE \t\t MedianAE \t SMAPE \t\t MASE \t\t NMAE \t\t WMAPE \t\t MALE', file=file)
    print('-------------------------------------------------------------------------------------------------------------------------------', file=file)
    spec_name_all = []
    medianae_all = []
    mae_all = []
    nmae_all = []
    smape_all = []
    wsmape_all = []
    male_all = []
    mase_all = []

    meatadata_frame_cat = metadata_frame.copy()
    meatadata_frame_cat['genus_species'] = meatadata_frame_cat.tree_genus + ' ' + meatadata_frame_cat.tree_species

    for species_name, subdf in meatadata_frame_cat.groupby('genus_species'):
        ser_ids = subdf.index.values
        mask = np.isin(data_dict['series_ids'], ser_ids)

        if filter_sids is not None:
            mask_filter = np.isin(data_dict['series_ids'], filter_sids)
        else:
            mask_filter = np.ones((data_dict['series_ids'].shape[0], ), dtype=np.bool_)

        mask = np.bitwise_and(mask, mask_filter)

        if mask.sum() > 0:
            forecasts_filt = data_dict['forecasts'][mask, :forecast_horizon, quant_idx]
            targets_filt = data_dict['fcast_tgts'][mask, :forecast_horizon, 0]
            naive_filt = data_dict['bcast_tgts'][mask, -1:, 0]
            forecasts_filt_clamp = torch.clamp(forecasts_filt, min=1e-6)
            targets_filt_clamp = torch.clamp(targets_filt, min=1e-6)
            naive_filt_clamp = torch.clamp(naive_filt, min=1e-6)

            medianae = torch.median(torch.abs(forecasts_filt_clamp - targets_filt_clamp))
            mae = torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp))
            nmae = torch.sum(torch.abs(forecasts_filt_clamp - targets_filt_clamp)) / torch.sum(torch.abs(targets_filt_clamp)) 
            smape = 100 * torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp) / (torch.abs(forecasts_filt_clamp) + torch.abs(targets_filt_clamp)))
            wsmape = 100 * torch.mean(torch.sum(torch.abs(forecasts_filt_clamp - targets_filt_clamp), dim=1) / torch.sum(torch.abs(forecasts_filt_clamp) + torch.abs(targets_filt_clamp), dim=1))
            male = torch.mean(torch.abs(torch.log(forecasts_filt_clamp / targets_filt_clamp)))
            mase = torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp) / (torch.mean(torch.abs(naive_filt_clamp - targets_filt_clamp), dim=-1, keepdims=True) + 1)) 
            print('%s \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (species_name.ljust(20), mae, medianae, smape, mase, nmae, wsmape, male), file=file)  
            
            spec_name_all.append(species_name)
            medianae_all.append(torch.median(torch.abs(forecasts_filt_clamp - targets_filt_clamp), dim=0)[0])
            mae_all.append(torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp), dim=0))
            nmae_all.append(torch.sum(torch.abs(forecasts_filt_clamp - targets_filt_clamp), dim=0) / (torch.sum(torch.abs(targets_filt_clamp), dim=0) + 1))
            smape_all.append(100 * torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp) / (torch.abs(forecasts_filt_clamp) + torch.abs(targets_filt_clamp)), dim=0))
            wsmape_all.append(100 * torch.mean(torch.sum(torch.abs(forecasts_filt_clamp - targets_filt_clamp), dim=1) / torch.sum(torch.abs(forecasts_filt_clamp) + torch.abs(targets_filt_clamp), dim=1), dim=0))
            male_all.append(torch.mean(torch.abs(torch.log(forecasts_filt_clamp / targets_filt_clamp)), dim=0))
            mase_all.append(torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp) / (torch.mean(torch.abs(naive_filt_clamp - targets_filt_clamp), dim=-1, keepdims=True) + 1), dim=0))

    spec_name_all = np.stack(spec_name_all)
    medianae_all = np.stack(medianae_all)
    mae_all = np.stack(mae_all)
    nmae_all = np.stack(nmae_all)
    smape_all = np.stack(smape_all)
    wsmape_all = np.stack(wsmape_all)
    male_all = np.stack(male_all)
    mase_all = np.stack(mase_all)

    return {
        'spec_name': spec_name_all,
        'medianae': medianae_all,
        'mae': mae_all,
        'nmae': nmae_all,
        'smape': smape_all,
        'wsmape': wsmape_all,
        'male': male_all,
        'mase': mase_all,
    }


def per_species_performance_paper(data_dict, metadata_frame, filter_sids : Optional[List[int]] = None, experiment_name : str = '', file=sys.stdout):  
    quant_idx = data_dict['forecasts'].shape[-1] // 2
    forecast_horizon = data_dict['forecasts'].shape[-2]
    
    print('Per-species data - %s' % experiment_name, file=file)
    print('-------------------------------------------------------------------------------------------------------------------------------', file=file)
    print('Species Name & Support & MAE & MedianAE & MASE', file=file)
    print('-------------------------------------------------------------------------------------------------------------------------------', file=file)

    meatadata_frame_cat = metadata_frame.copy()
    meatadata_frame_cat['genus_species'] = meatadata_frame_cat.tree_genus + ' ' + meatadata_frame_cat.tree_species

    for species_name, subdf in meatadata_frame_cat.groupby('genus_species'):
        ser_ids = subdf.index.values
        mask = np.isin(data_dict['series_ids'], ser_ids)

        if filter_sids is not None:
            mask_filter = np.isin(data_dict['series_ids'], filter_sids)
        else:
            mask_filter = np.ones((data_dict['series_ids'].shape[0], ), dtype=np.bool_)

        mask = np.bitwise_and(mask, mask_filter)

        if mask.sum() > 0:
            forecasts_filt = data_dict['forecasts'][mask, :forecast_horizon, quant_idx]
            targets_filt = data_dict['fcast_tgts'][mask, :forecast_horizon, 0]
            naive_filt = data_dict['bcast_tgts'][mask, -1:, 0]
            forecasts_filt_clamp = torch.clamp(forecasts_filt, min=1e-6)
            targets_filt_clamp = torch.clamp(targets_filt, min=1e-6)
            naive_filt_clamp = torch.clamp(naive_filt, min=1e-6)

            medianae = torch.median(torch.abs(forecasts_filt_clamp - targets_filt_clamp))
            mae = torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp))
            mase = torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp) / (torch.mean(torch.abs(naive_filt_clamp - targets_filt_clamp), dim=-1, keepdims=True) + 1)) 
            print('%s & %i & %.4f & %.4f & %.4f \\\\' % (species_name, mask.sum(), mae, medianae,  mase), file=file)  
        
def per_sample_performance(data_dict, data_mask, sample_idx, file=sys.stdout) -> Tuple[float, float, float]:  
    quant_idx = data_dict['forecasts'].shape[-1] // 2
    forecast_horizon = data_dict['forecasts'].shape[-2]
    
    forecasts_filt = data_dict['forecasts'][data_mask, :forecast_horizon, quant_idx][sample_idx, :]
    targets_filt = data_dict['fcast_tgts'][data_mask, :forecast_horizon, 0][sample_idx, :]
    naive_filt = data_dict['bcast_tgts'][data_mask, -1:, 0][sample_idx, :]
    forecasts_filt_clamp = torch.clamp(forecasts_filt, min=1e-6)
    targets_filt_clamp = torch.clamp(targets_filt, min=1e-6)
    naive_filt_clamp = torch.clamp(naive_filt, min=1e-6)

    medianae = torch.median(torch.abs(forecasts_filt_clamp - targets_filt_clamp))
    mae = torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp))
    mase = torch.mean(torch.abs(forecasts_filt_clamp - targets_filt_clamp) / (torch.mean(torch.abs(naive_filt_clamp - targets_filt_clamp), dim=-1, keepdims=True) + 1)) 
    
    return mae, medianae, mase
        