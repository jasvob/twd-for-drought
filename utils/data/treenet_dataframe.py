import os
import pandas as pd
import numpy as np
from typing import List, Callable, Optional, Tuple, Optional

class TreeNetDataFrame(object):
    """
    Dataframe class to handle data from the TreeNet network.
    Internally it holds an instance of pandas dataframe and offers functionality for data filtering and preprocessing in addition.

    Parameters
    ---------- 
        data_file_path : str
            Path to the Parquet with TreeNet station data.

        metadata_filter_fn : str
            Function that applies additional filtering to the metadata frame. Defaults to ''.

        treedata_filter_fn : str
            Function that applies additional filtering to each tree's the data frame. Defaults to ''.

        data_filter_fn : str
            Post filtering of the final merged data frame. Defaults to ''.

        min_trees_per_site : int
            Minimum number of trees at each site. Defaults to 2.

        num_trees_per_date : Optional[int]
            Required number of trees at each date. Defaults to None.

        equal_sequence_per_site : bool
            Whether each site has to contain data point only available for all trees in the site. Defaults to False.

        interpolate_data : bool
            Whether to interpolate missing values in the data. Defaults to False.

        drop_na_for_variables : Optional[List[str]]
            Whether to drop NaN values for variables selected in the list given in this paramter. Optional parameter, defaults to None.

        drop_na_for_metadata : Optional[List[str]]
            Whether to drop NaN values for metadata selected in the list given in this paramter. Optional parameter, defaults to None.

        exclude_series : Optional[List[int]]
            Optional list of series ids to exclude from the dataset. Defaults to None.

        resample_data  : Optional[Tuple[str, str]] = None
            Optional datetime resampling string and offset string for the data. If provided, data is resampled into the given frequency, for example '1H', '1d', '30min'. Optional parameter, defaults to None.

        order_by : List[str]
            List of column names to order the final dataframe by. Defaults to ['site_id', 'ts', 'series_id'].

        filter_growth_period : bool 
            Whether to filter tree data by growth period. Defaults to False.

        filter_min_growth : Optional[float]
            Filter minimum growth per year. Optional, defaults to None.

        filter_min_twd : Optional[Tuple[float, float]]
            Filter minimum twd per year. Tuple of two floats, first one is the quantile, second is the minimum TWD for that quantile. Optional, defaults to None.
          
        merge_by_site : Optional[str]
            Function to use to merge data by site IDs. Defaults to None, which means do not merge.
    """

    def __init__(self, data_file_path : str, metadata_file_path : str, metadata_filter_fn : str = '', treedata_filter_fn : str = '', data_filter_fn : str = '', cache_path : Optional[str] = None,  min_trees_per_site : int = 2, num_trees_per_date : Optional[int] = None, equal_sequence_per_site : bool = False, interpolate_data : bool = False, drop_na_for_variables : Optional[List[str]] = None, drop_na_for_metadata : Optional[List[str]] = None, exclude_series : Optional[List[int]] = None, resample_data : Optional[Tuple[str, str]] = None, order_by : List[str] = ['site_id', 'ts', 'series_id'], filter_growth_period : bool = False, filter_by_temperature : Optional[Tuple[str, float]] = None, filter_by_frost : bool = False, filter_min_growth : Optional[float] = None, filter_min_twd : Optional[Tuple[float, float]] = None, merge_by_site : Optional[str] = None) -> None:
        super(TreeNetDataFrame, self).__init__()

        # If we have the dataframe in cache, reload it
        if cache_path is not None and os.path.exists(cache_path + '_df.parquet') and os.path.exists(cache_path + '_metadf.parquet'):
            print('Loading generated data frames from cache...')
            self.data_frames = pd.read_parquet(cache_path + '_df.parquet')

            if len(data_filter_fn) > 0:
                self.data_frames = self.data_frames.query(data_filter_fn)
                
            self.site_ids = self.data_frames.index.get_level_values('site_id').unique()
            self.measure_dates = self.data_frames.index.get_level_values('ts').unique()
            self.metadata_frame = pd.read_parquet(cache_path +  '_metadf.parquet')
            self.site_metadata_frame = self.metadata_frame.drop_duplicates(subset=['site_id'], keep='first').set_index('site_id')
            print('Done.')
        # otherwise initialize from scratch
        else:
            if cache_path is not None:
                print('Cache path given, but cache files do not exists.')
            else:
                print('Cache path not given.')
            print('Generating data frames from scratch...')
            # Read metadata
            metadf = pd.read_parquet(metadata_file_path)
            # Add metadata genus and species ID
            all_genus = metadf['tree_genus'].drop_duplicates()
            all_genus = {gen : idx for idx, gen in enumerate(all_genus)}
            metadf['genus_id'] = metadf['tree_genus'].map(lambda x: all_genus[x])
            all_species = metadf['tree_species'].drop_duplicates()
            all_species = {spec : idx for idx, spec in enumerate(all_species)}
            metadf['species_id'] = metadf['tree_species'].map(lambda x: all_species[x])

            # Apply metadata data filter if provided
            if len(metadata_filter_fn) > 0:
                metadf = metadf.query(metadata_filter_fn)

            # Read data from parquet file
            df_all = []
            for site_id in metadf['site_id'].unique():
                site_series_ids = metadf[metadf['site_id'] == site_id]['series_id'].to_numpy()

                if min_trees_per_site <= site_series_ids.shape[0]:
                    df_site_trees = []
                    for series_id in metadf[metadf['site_id'] == site_id]['series_id'].to_numpy():
                        print('Processing series %s...' % series_id)
                        
                        # In case this series ID is in the exclude list, skip it
                        if exclude_series is not None and series_id in exclude_series:
                            continue
                        
                        ###print(series_id, site_id)
                        df_tree = pd.read_parquet(os.path.join(data_file_path, 'data_%04i_%04i.parquet' % (site_id, series_id)))
                        df_tree = df_tree.reset_index()
                        
                        # Make sure values are sorted chronologically
                        df_tree = df_tree.sort_values(by='ts')

                        # Get tree metadata
                        tree_metadata = metadf[metadf['series_id'] == series_id]
                        site_lon = tree_metadata['site_xcor'].apply(pd.to_numeric).values[0]
                        site_lat = tree_metadata['site_ycor'].apply(pd.to_numeric).values[0]

                        # Define yearly dendrometer signal using TWD and gro_yr
                        df_tree['value_yr'] = df_tree['gro_yr'].apply(pd.to_numeric) - df_tree['twd'].apply(pd.to_numeric)
                        df_tree['gro_hr'] = df_tree['gro_yr'].apply(pd.to_numeric).diff().clip(lower=0.0)
                        df_tree.loc[df_tree['gro_hr'] > tree_metadata['series_gro_max_hr'].apply(pd.to_numeric).values[0], ['gro_hr']] = 0.0
                        df_tree['genus_id'] = tree_metadata['genus_id'].values[0]
                        df_tree['species_id'] = tree_metadata['species_id'].values[0]
                        df_tree['site_id'] = tree_metadata['site_id'].values[0]

                        # Compute evapotranspiration using empirical formula based on temperature and solar radiation
                        # Don't forget to convert degrees Celsius to Fahrenheit as the formula expects Fahrenheit
                        df_tree['evt'] = (0.014 * (df_tree['temp'].apply(pd.to_numeric) + 32) - 0.37) * df_tree['rad'].apply(pd.to_numeric)

                        # Create normalized TWD
                        if tree_metadata['series_mds_gp_max'].values[0] is not None:
                            df_tree['twd_norm'] = df_tree['twd'].apply(pd.to_numeric) / tree_metadata['series_mds_gp_max'].apply(pd.to_numeric).values[0]
                            df_tree['twd_norm'] = df_tree['twd_norm'].clip(lower=0.0, upper=5.0) # Realistically, maximum TWD norm value can be around 5.0
                            df_tree['value_yr_norm'] = df_tree['value_yr'].apply(pd.to_numeric) / tree_metadata['series_mds_gp_max'].apply(pd.to_numeric).values[0]
                        else:
                            if tree_metadata['series_gro_start_doy_med'].isna().sum() == 0 and tree_metadata['series_gro_end_doy_med'].isna().sum() == 0:
                                start_of_growth = int(tree_metadata['series_gro_start_doy_med'].values[0])
                                end_of_growth = int(tree_metadata['series_gro_end_doy_med'].values[0])
                                df_tree_gro = df_tree.query('ts.dt.day_of_year >= %i and ts.dt.day_of_year <= %i' % (start_of_growth, end_of_growth))
                                
                                df_tree_gro = df_tree_gro.set_index('ts')
                                df_day_min = df_tree_gro['value'].resample('16H', offset='0H').min()
                                df_day_max = df_tree_gro['value'].resample('16H', offset='0H').max()

                                df_day_diff = (df_day_max - df_day_min)
                                mean_diff = df_day_diff.mean()
                                stddev_diff = df_day_diff.apply(pd.to_numeric).std()
                                diff_filt = df_day_diff < (mean_diff + stddev_diff * 1.5)
                                
                                df_tree['twd_norm'] = df_tree['twd'].apply(pd.to_numeric) / df_day_diff[diff_filt].apply(pd.to_numeric).max()
                                df_tree['twd_norm'] = df_tree['twd_norm'].clip(lower=0.0, upper=5.0) # Realistically, maximum TWD norm value can be around 5.0
                                df_tree['value_yr_norm'] = df_tree['value_yr'].apply(pd.to_numeric) / df_day_diff[diff_filt].apply(pd.to_numeric).max()
                            else:
                                continue

                        # If requested, linearly interpolate missing data
                        if interpolate_data:
                            df_int = df_tree.set_index('ts')
                            df_int = df_int.resample('1h')
                            df_int = df_int.interpolate(method='linear')
                            df_int = df_int.reset_index()
                            df_tree = df_int

                        if resample_data is not None:
                            df_int = df_tree.set_index('ts')
                            df_int_num = df_int.select_dtypes('number').resample(resample_data[0], offset=resample_data[1]).mean() # (mean, sum, median, max, min)
                            df_int_bool = df_int.select_dtypes('bool').resample(resample_data[0], offset=resample_data[1]).max() # (mean, sum, median, max, min)
                            df_int_obj = df_int.select_dtypes('object').resample(resample_data[0], offset=resample_data[1]).first() # (mean, sum, median, max, min)
                            df_int_precip = df_int['total_precip'].resample(resample_data[0], offset=resample_data[1]).sum() 
                            df_int_twd = df_int['twd'].resample(resample_data[0], offset=resample_data[1]).min() 
                            df_int_twd_norm = df_int['twd_norm'].resample(resample_data[0], offset=resample_data[1]).min() 
                            df_int_gro_hr = df_int['gro_hr'].resample(resample_data[0], offset=resample_data[1]).sum() 
                            df_int = df_int_num.join(df_int_obj).join(df_int_bool)
                            df_int['total_precip'] = df_int_precip
                            df_int['twd'] = df_int_twd
                            df_int['twd_norm'] = df_int_twd_norm
                            df_int['gro_hr'] = df_int_gro_hr
                            df_int = df_int.reset_index()
                            df_tree = df_int
                        
                        doy_enc = (df_tree.ts.dt.day_of_year / 366.0) * np.pi * 2.0
                        df_tree['doy_encoding'] = np.sin(doy_enc) + np.cos(doy_enc)

                        df_tree['doy'] = df_tree.ts.dt.day_of_year
                        df_tree['moy'] = df_tree.ts.dt.month
                        df_tree['hod'] = df_tree.ts.dt.hour

                        # Compute growth period flag
                        if tree_metadata['series_gro_start_doy_med'].isna().sum() == 0 and tree_metadata['series_gro_end_doy_med'].isna().sum() == 0:
                            start_of_growth = int(tree_metadata['series_gro_start_doy_med'].values[0])
                            end_of_growth = int(tree_metadata['series_gro_end_doy_med'].values[0])
                            df_tree['growth_period'] = np.bitwise_and(df_tree.ts.dt.day_of_year >= start_of_growth, df_tree.ts.dt.day_of_year <= end_of_growth)
                        else: # If growth period filter is required but the data about growth period are not available, discard the tree
                            continue

                        # Filter only data that fall into growth period if required
                        if filter_growth_period:
                            if tree_metadata['series_gro_start_doy_med'].isna().sum() == 0 and tree_metadata['series_gro_end_doy_med'].isna().sum() == 0:
                                start_of_growth = int(tree_metadata['series_gro_start_doy_med'].values[0])
                                end_of_growth = int(tree_metadata['series_gro_end_doy_med'].values[0])
                                df_tree = df_tree.query('ts.dt.day_of_year >= %i and ts.dt.day_of_year <= %i' % (start_of_growth, end_of_growth))
                            else: # If growth period filter is required but the data about growth period are not available, discard the tree
                                continue

                        # If filter by tempreature is provided, first value is the sign of the comparison and second is the temperature threshold
                        if filter_by_temperature is not None: 
                            df_tree = df_tree.query('temp %s %f' % (filter_by_temperature[0], filter_by_temperature[1]))

                        if filter_by_frost: 
                            df_tree = df_tree.query('frost == False')

                        if filter_min_growth is not None:
                            df_yr = df_tree[['ts', 'gro_yr']].set_index('ts').resample('1Y').max()
                            gro_years = df_yr[df_yr.gro_yr > filter_min_growth].reset_index().ts.dt.year.tolist()
                            df_tree = df_tree.iloc[np.where(df_tree.ts.dt.year.isin(gro_years))[0]]
                            if df_tree.shape[0] <= 0:
                                continue
                        
                        if filter_min_twd is not None:
                            df_yr = df_tree[['ts', 'twd']].set_index('ts').apply(pd.to_numeric).resample('1Y').quantile(q=filter_min_twd[0])
                            twd_years = df_yr[df_yr.twd > filter_min_twd[1]].reset_index().ts.dt.year.tolist()
                            df_tree = df_tree.iloc[np.where(df_tree.ts.dt.year.isin(twd_years))[0]]
                            if df_tree.shape[0] <= 0:
                                continue

                        # Apply data filter if provided
                        if len(treedata_filter_fn) > 0:
                            df_tree = df_tree.query(treedata_filter_fn)

                        if drop_na_for_variables is not None:
                            df_tree = df_tree.dropna(subset=drop_na_for_variables)

                        if drop_na_for_metadata is not None:
                            if tree_metadata[drop_na_for_metadata].isna().to_numpy().sum() > 0:
                                continue
                        
                        if len(df_tree) <= 1:
                            continue

                        # Add transformed variables
                        df_tree['total_precip_log1p'] = df_tree['total_precip'].apply(pd.to_numeric).apply(np.log1p)
                        df_tree['swp_diff'] = df_tree['swp'].apply(pd.to_numeric).diff().fillna(0)
                        df_tree['swp_pct_change'] = df_tree['swp'].apply(pd.to_numeric).pct_change().fillna(0)
                        df_tree['swp_abs_log1p'] = df_tree['swp'].abs().apply(pd.to_numeric).apply(np.log1p)
                        df_tree['swp_log10_pf'] = df_tree['swp'].apply(pd.to_numeric).apply(lambda x: np.log10(np.abs(x / 100)))
                        df_tree['rad_log1p'] = df_tree['rad'].apply(pd.to_numeric).apply(np.log1p)
                        df_tree['swp_diff'] = df_tree['swp'].apply(pd.to_numeric).diff().fillna(0)
                        df_tree['swp_log1p_diff'] = df_tree['swp'].apply(pd.to_numeric).abs().apply(np.log1p).diff().fillna(0)
                        df_tree['swp_log1p_pct_change'] = df_tree['swp'].apply(pd.to_numeric).abs().apply(np.log1p).pct_change().fillna(0)

                        df_tree['year'] = df_tree['ts'].dt.year
                        df_tree['twd_norm_sigmax'] = df_tree['twd_norm'].apply(pd.to_numeric) / df_tree['twd_norm'].apply(pd.to_numeric).max()
                        df_tree['twd_norm_yearmax'] = df_tree['twd_norm'].apply(pd.to_numeric) / df_tree.groupby('year')['twd_norm'].transform('max').apply(pd.to_numeric)

                        if tree_metadata['series_gro_start_doy_med'].isna().sum() == 0 and tree_metadata['series_gro_end_doy_med'].isna().sum() == 0:
                            start_of_growth = int(tree_metadata['series_gro_start_doy_med'].values[0])
                            end_of_growth = int(tree_metadata['series_gro_end_doy_med'].values[0])
                            df_tree_gro = df_tree.query('ts.dt.day_of_year >= %i and ts.dt.day_of_year <= %i' % (start_of_growth, end_of_growth))
                            df_tree['twd_norm_gromax'] = df_tree['twd_norm'].apply(pd.to_numeric) / df_tree_gro['twd_norm'].apply(pd.to_numeric).max()
                        else: 
                            df_tree['twd_norm_gromax'] = df_tree['twd_norm'].apply(pd.to_numeric) / df_tree['twd_norm'].apply(pd.to_numeric).max()
                        # Clip winter TWD if it was highar than max. growth period TWD
                        df_tree['twd_norm_gromax'] = df_tree['twd_norm_gromax'].clip(lower=0.0, upper=1.0)

                        df_site_trees.append(df_tree)

                    num_trees = len(df_site_trees)
                    # If there are no trees for this site, continue with the next site
                    if num_trees == 0:
                        continue
                    
                    df_site_trees = pd.concat(df_site_trees)
                    if equal_sequence_per_site:
                        # Merge trees back into a subplot
                        # Make sure to keep only records available for all trees in the subplot
                        trees_for_date = df_site_trees.groupby(['ts']).count().reset_index()
                        valid_ts = trees_for_date[trees_for_date['series_id'] == num_trees]['ts']
                        df_site_trees = df_site_trees.set_index('ts')
                        df_site_trees = df_site_trees.loc[valid_ts]
                        df_site_trees = df_site_trees.reset_index()
                        df_site_trees = df_site_trees.drop(columns=['index'])
                        
                    # Add into a list of dataframes of all generated 'sites'
                    df_all.append(df_site_trees)

            # Merge everything into one single big dataframe 
            # Thanks to the new columns 'site_id' and 'series_id', different trees in various sites can be quickly recovered
            df_all = pd.concat(df_all)

            if num_trees_per_date is not None:
                # Get number of records per series ID
                sid_counts = df_all.groupby(['series_id']).count()
                sid_counts = sid_counts.sort_values(by=['value'])['value'].reset_index()
            
                # Filter only the num_trees_per_date trees that have the most records
                sids = sid_counts['series_id'][-num_trees_per_date:].to_list()
                df_all = df_all[df_all.series_id.isin(sids)]

                # Make sure to keep only the records which are available for all trees in the dataframe
                avail_for_all = df_all.groupby(['ts']).count() == num_trees_per_date
                avail_for_all = avail_for_all.reset_index()
                df_all = df_all[df_all.ts.isin(avail_for_all[avail_for_all['series_id'] == True].ts)]
            
            self.data_frames = df_all.sort_values(by=order_by)

            # If merging by site is desired, merge by site
            if merge_by_site is None:   
                self.metadata_frame = metadf
                self.metadata_frame = self.metadata_frame.set_index('series_id')
                self.metadata_frame['series_id'] = self.metadata_frame.index.values
                self.site_metadata_frame = metadf.drop_duplicates(subset=['site_id'], keep='first').set_index('site_id')
            else:
                self.metadata_frame = metadf
                self.metadata_frame = self.metadata_frame.groupby(['site_id', 'genus_id', 'species_id']).first()

                self.data_frames = self.data_frames.groupby(['ts', 'site_id', 'genus_id', 'species_id']).agg(
                    {
                        'index' : 'first', 'gro_yr' : merge_by_site, 'gro_hr': merge_by_site, 'value_yr' : merge_by_site, 'value_yr_norm' : merge_by_site, 'value' : merge_by_site, 'max' : merge_by_site, 'twd' : merge_by_site,
                        'gro_start' : 'min', 'gro_end' : 'max', 'flags' : 'first', 'growth_period': 'max', 'frost': 'max', 'temp' : merge_by_site, 'rh' : merge_by_site, 'swp' : merge_by_site, 'total_precip' : merge_by_site,
                        'rad' : merge_by_site, 'vpd' : merge_by_site, 'vpd_bo' : merge_by_site, 'twd_norm' : merge_by_site, 'twd_norm_sigmax' : merge_by_site, 'twd_norm_gromax' : merge_by_site,  'twd_norm_yearmax' : merge_by_site, 
                        'doy_encoding' : merge_by_site, 'doy' : merge_by_site, 'moy' : merge_by_site, 'hod' : merge_by_site, 'evt' : merge_by_site,
                        'total_precip_log1p' : merge_by_site, 'swp_diff' : merge_by_site, 'swp_pct_change' : merge_by_site, 'swp_abs_log1p' : merge_by_site,
                        'swp_log10_pf' : merge_by_site, 'rad_log1p' : merge_by_site, 'swp_log1p_diff' : merge_by_site, 'swp_log1p_pct_change' : merge_by_site, 'year' : merge_by_site
                    }
                )
                self.data_frames = self.data_frames.reset_index()
                self.data_frames['series_id'] = self.data_frames.apply(lambda x: self.metadata_frame.loc[(x['site_id'], x['genus_id'], x['species_id'])]['series_id'], axis=1)
                
                self.metadata_frame = self.metadata_frame.reset_index()
                self.metadata_frame = self.metadata_frame.set_index('series_id')
                self.metadata_frame['series_id'] = self.metadata_frame.index.values
                self.site_metadata_frame = metadf.drop_duplicates(subset=['site_id'], keep='first').set_index('site_id')

            if len(data_filter_fn) > 0:
                self.data_frames = self.data_frames.query(data_filter_fn)

            # Sort data frame and set its index
            self.data_frames = self.data_frames.sort_values(by=order_by)
            self.data_frames = self.data_frames.set_index(keys=['site_id', 'ts', 'series_id'])
                
            # Extract index unique values
            self.measure_dates = self.data_frames.index.get_level_values('ts').unique()
            self.site_ids = self.data_frames.index.get_level_values('site_id').unique().sort_values()

            # If cache path is given, save data frame to cache
            if cache_path is not None:
                print('Saving generated data frames to cache...')
                self.data_frames.to_parquet(cache_path + '_df.parquet')
                self.metadata_frame.to_parquet(cache_path + '_metadf.parquet')
                print('Done.')

        print(self.data_frames.shape)

    def __len__(self) -> int:
        return len(self.data_frames)

    def get_treenet_datetime_data_frame(self, datetime : pd.Timestamp) -> pd.DataFrame:
        return self.data_frames.loc[(slice(None), datetime, slice(None)), :]
    
    def get_treenet_datetime_sequence_data_frame(self, datetime_from : pd.Timestamp, datetime_to : pd.Timestamp) -> pd.DataFrame:
        return self.data_frames.loc[(slice(None), slice(datetime_from, datetime_to), slice(None)), :]

    def get_site_data_frame(self, site_id : int) -> pd.DataFrame:
        indexer = self.data_frames.index.isin([site_id], level='site_id')
        return self.data_frames[indexer]
    
    def get_site_metadata(self, site_id : int) -> pd.DataFrame:
        return self.metadata_frame[self.metadata_frame['site_id'] == site_id]
    
    def get_tree_datetime_data_frame(self, site_id : int, series_id : int, datetime : pd.Timestamp) -> pd.DataFrame:
        return self.data_frames.loc[(site_id, datetime, series_id), :]
    
    def get_tree_datetime_sequence_data_frame(self, site_id : int, series_id : int, datetime_from : pd.Timestamp, datetime_to : pd.Timestamp) -> pd.DataFrame:
        return self.data_frames.loc[(site_id, slice(datetime_from, datetime_to), series_id), :]
    
    def get_site_datetime_data_frame(self, site_id : int, datetime : pd.Timestamp) -> pd.DataFrame:
        return self.data_frames.loc[(site_id, datetime, slice(None)), :]
    
    def get_site_datetime_sequence_data_frame(self, site_id : int, datetime_from : pd.Timestamp, datetime_to : pd.Timestamp) -> pd.DataFrame:
        return self.data_frames.loc[(site_id, slice(datetime_from, datetime_to), slice(None)), :]

    def get_site_ids(self) -> List[int]:
        return list(self.site_ids)
    
    def get_tree_metadata(self, series_id : int) -> pd.DataFrame:
        return self.metadata_frame.loc[series_id]

    def get_tree_list_metadata(self, series_ids : List[int]) -> pd.DataFrame:
        return self.metadata_frame.loc[series_ids]

    def get_available_datetimes(self) -> List[pd.Timestamp]:
        return self.measure_dates

