import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset
from typing import Tuple, List, Optional

from utils.data.treenet_dataframe import TreeNetDataFrame
from ml.data.misc import get_sequence_indices_treenet_tree

class TreeNetDataset(Dataset):
    """
    Basic PyTorch Dataset class to handle data from and TreeNet stations.

    Parameters
    ----------    
        treenet_data_frame : TreeNetDataFrame
            TreeNetDataFrame class this dataset should take data from.

        input_feats : List[str]
            List of names of the columns to take from the dataframe as input features.

        exog_feats : List[str]
            List of names of the columns to take from the metadataframe as exogenous features.

        target_feats : List[str]
            Name of the column to take as the target feature.

        normalization : Tuple[str, str, str] 
            Type of normalization to be used on inputs and targets. First element of the tuple specifies type of inputs, the second for targets.
            One of ['none', 'meanstd', 'minmax']. Defaults to ['none', 'none'].

        norm_inputs : Tuple[List[float], List[float]]
            Normalization constants for the input variables according to the chosen normalization type. 
            If None, the normalization constants are computed automatically from the data. Defaults to None.

        norm_exog : Tuple[List[float], List[float]]
            Normalization constants for the exogenous variables according to the chosen normalization type. 
            If None, the normalization constants are computed automatically from the data. Defaults to None.

        norm_targets : Tuple[List[float], List[float]]
            Normalization constants for the target variable according to the chosen normalization type. 
            If None, the normalization constants are computed automatically from the data. Defaults to None.

        return_timestamp : bool
            Whether to return the timestamp for each sample or not. Defaults to False.
    """

    def __init__(self, treenet_data_frame : TreeNetDataFrame, input_feats : List[str], exog_feats : List[str], target_feats : List[str], normalization : Tuple[str, str, str] = ('none', 'none', 'none'), norm_inputs : Tuple[List[float], List[float]] = None, norm_exog : Tuple[List[float], List[float]] = None, norm_targets : Tuple[List[float], List[float]] = None, target_only_growth : Optional[int] = None, return_timestamp : bool = False) -> None:
        super(TreeNetDataset, self).__init__()

        self.treenet_data_frame = treenet_data_frame

        self.input_feats = input_feats.copy()
        self.exog_feats = exog_feats
        self.target_feats = target_feats
        self.normalization = normalization
        self.norm_inputs = norm_inputs
        self.norm_exog = norm_exog
        self.norm_targets = norm_targets
        self.target_only_growth = target_only_growth
        self.return_timestamp = return_timestamp

        if self.norm_inputs  is not None:
            self.norm_inputs = (
                torch.tensor(norm_inputs[0], dtype=torch.float32),
                torch.tensor(norm_inputs[1], dtype=torch.float32)
            )
        if self.norm_exog  is not None:
            self.norm_exog = (
                torch.tensor(norm_exog[0], dtype=torch.float32),
                torch.tensor(norm_exog[1], dtype=torch.float32)
            )
        if self.norm_targets  is not None:
            self.norm_targets = (
                torch.tensor(norm_targets[0], dtype=torch.float32),
                torch.tensor(norm_targets[1], dtype=torch.float32)
            )

        # Compute normalization values 
        if self.norm_inputs is None:
            if self.normalization[0] == 'meanstd':
                self.norm_inputs = (
                    torch.tensor(self.treenet_data_frame.data_frames[self.input_feats].apply(pd.to_numeric).to_numpy().astype(np.float32).mean(axis=0), dtype=torch.float32),
                    torch.tensor(self.treenet_data_frame.data_frames[self.input_feats].apply(pd.to_numeric).to_numpy().astype(np.float32).std(axis=0), dtype=torch.float32)
                )
            elif self.normalization[0] == 'minmax':
                self.norm_inputs = (
                    torch.tensor(self.treenet_data_frame.data_frames[self.input_feats].apply(pd.to_numeric).to_numpy().astype(np.float32).min(axis=0), dtype=torch.float32),
                    torch.tensor(self.treenet_data_frame.data_frames[self.input_feats].apply(pd.to_numeric).to_numpy().astype(np.float32).max(axis=0), dtype=torch.float32)
                )
        if self.norm_exog is None:
            if self.normalization[1] == 'meanstd':
                self.norm_exog = (
                    torch.tensor(self.treenet_data_frame.metadata_frame[self.exog_feats].apply(pd.to_numeric).mean(axis=0).to_numpy().astype(np.float32), dtype=torch.float32),
                    torch.tensor(self.treenet_data_frame.metadata_frame[self.exog_feats].apply(pd.to_numeric).std(axis=0).to_numpy().astype(np.float32), dtype=torch.float32)
                )
            elif self.normalization[1] == 'minmax':
                self.norm_exog = (
                    torch.tensor(self.treenet_data_frame.metadata_frame[self.exog_feats].apply(pd.to_numeric).min(axis=0).to_numpy().astype(np.float32), dtype=torch.float32),
                    torch.tensor(self.treenet_data_frame.metadata_frame[self.exog_feats].apply(pd.to_numeric).max(axis=0).to_numpy().astype(np.float32), dtype=torch.float32)
                )
        if self.norm_targets is None:
            if self.normalization[2] == 'meanstd':
                self.norm_targets = (
                    torch.tensor(self.treenet_data_frame.data_frames[self.target_feats].apply(pd.to_numeric).to_numpy().astype(np.float32).mean(), dtype=torch.float32),
                    torch.tensor(self.treenet_data_frame.data_frames[self.target_feats].apply(pd.to_numeric).to_numpy().astype(np.float32).std(), dtype=torch.float32)
                )
            elif self.normalization[2] == 'minmax':
                self.norm_targets = (
                    torch.tensor(self.treenet_data_frame.data_frames[self.target_feats].apply(pd.to_numeric).to_numpy().astype(np.float32).min(), dtype=torch.float32),
                    torch.tensor(self.treenet_data_frame.data_frames[self.target_feats].apply(pd.to_numeric).to_numpy().astype(np.float32).max(), dtype=torch.float32)
                )

        self.df_indexer = self.treenet_data_frame.data_frames.reset_index().sort_values(by=['site_id', 'series_id', 'ts'])[['site_id', 'ts', 'series_id', 'growth_period']].reset_index()

    def __len__(self) -> int:
        return len(self.df_indexer)
    
    def normalize_data(self, inputs : torch.Tensor, norm_type : str, norm_constants : Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if norm_type == 'meanstd':
            return (inputs - norm_constants[0]) / norm_constants[1]
        elif norm_type == 'minmax':
            return (inputs - norm_constants[0]) / (norm_constants[1] - norm_constants[0])
        elif norm_type == 'none':
            return inputs
        else:
            raise NotImplementedError("Unknown data normalization type requested.")

    def unnormalize_data(self, inputs : torch.Tensor, norm_type : str, norm_constants : Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if norm_type == 'meanstd':
            return inputs * norm_constants[1] + norm_constants[0]
        elif norm_type == 'minmax':
            return inputs * (norm_constants[1] - norm_constants[0]) + norm_constants[0]
        elif norm_type == 'none':
            return inputs
        else:
            raise NotImplementedError("Unknown data normalization type requested.")

    def __getitem__(self, index : int) -> Tuple[torch.Tensor]:
        """
        Convert a row of data from the Pandas dataframe into PyTorch tensor and return.
        
        Parameters
        ----------
            index : int
                Row (a datasample) of the dataframe to be processed.
        
        Returns
        -------
            (torch.Tensor, torch.Tensor) : Tuple of tensors where the first one represents input and second one the target.
        """

        # Get index
        index_datetime = self.df_indexer.iloc[index]['ts']
        index_site_id = self.df_indexer.iloc[index]['site_id']
        index_series_id = self.df_indexer.iloc[index]['series_id']

        df_tree = self.treenet_data_frame.get_tree_datetime_data_frame(index_site_id, index_series_id, index_datetime)
        df_tree_meta = self.treenet_data_frame.get_tree_metadata(index_series_id)

        inps = torch.tensor(df_tree.loc[self.input_feats].apply(pd.to_numeric).to_numpy(), dtype=torch.float32)
        inps = self.normalize_data(inps, self.normalization[0], self.norm_inputs)

        exog = torch.tensor(df_tree_meta.loc[self.exog_feats].apply(pd.to_numeric).to_numpy(), dtype=torch.float32)
        exog = self.normalize_data(exog, self.normalization[1], self.norm_exog)

        tgts = torch.tensor(df_tree.loc[self.target_feats].apply(pd.to_numeric).to_numpy(), dtype=torch.float32)
        tgts = self.normalize_data(tgts, self.normalization[2], self.norm_targets)

        if self.return_timestamp:
            tstamp = torch.tensor(index_datetime.value // 10 ** 9)
            return inps, exog, tgts, index_site_id, index_series_id, tstamp
        else:
            return inps, exog, tgts


class TreeNetTemporalDataset(TreeNetDataset):
    """
    Basic PyTorch Dataset class to handle temporal data from and TreeNet stations.

    Parameters
    ----------    
        treenet_data_frame : TreeNetDataFrame
            TreeNetDataFrame class this dataset should take data from.

        input_feats : List[str]
            List of names of the columns to take from the dataframe as input features.

        exog_feats : List[str]
            List of names of the columns to take from the metadataframe as exogenous features.

        target_feats : List[str]
            Name of the column to take as the target feature.

        seq_len : int
            Desired length of each sequence.
            
        max_dist_timesteps : int
            Maximum distance between the start and the end of the sequence in timesteps.

        normalization : Tuple[str, str, str] 
            Type of normalization to be used on inputs and targets. First element of the tuple specifies type of inputs, the second for targets.
            One of ['none', 'meanstd', 'minmax']. Defaults to ['none', 'none'].

        norm_inputs : Tuple[List[float], List[float]]
            Normalization constants for the input variables according to the chosen normalization type. 
            If None, the normalization constants are computed automatically from the data. Defaults to None.

        norm_exog : Tuple[List[float], List[float]]
            Normalization constants for the exogenous variables according to the chosen normalization type. 
            If None, the normalization constants are computed automatically from the data. Defaults to None.

        norm_targets : Tuple[List[float], List[float]]
            Normalization constants for the target variable according to the chosen normalization type. 
            If None, the normalization constants are computed automatically from the data. Defaults to None.

        return_timestamp : bool
            Whether to return the timestamp for each sample or not. Defaults to False.
    """

    def __init__(self, treenet_data_frame : TreeNetDataFrame, input_feats : List[str], exog_feats : List[str],  target_feats : List[str], seq_len : int, max_dist_timesteps : int, normalization : Tuple[str, str, str] = ('none', 'none', 'none'), norm_inputs : Tuple[List[float], List[float]] = None, norm_exog : Tuple[List[float], List[float]] = None, norm_targets : Tuple[List[float], List[float]] = None, target_only_growth : Optional[int] = None, return_timestamp : bool = False) -> None:
        super(TreeNetTemporalDataset, self).__init__(treenet_data_frame, input_feats, exog_feats, target_feats, normalization, norm_inputs, norm_exog, norm_targets, target_only_growth, return_timestamp)

        self.seq_len = seq_len
        self.max_dist_timesteps = max_dist_timesteps

        # Use this data frame to get the sequence indices
        self.sequence_idxs = get_sequence_indices_treenet_tree(self.df_indexer, window_size=seq_len, max_dist=max_dist_timesteps, index_filter=self.df_indexer.index.to_list(), target_only_growth=self.target_only_growth)          

    def __len__(self) -> int:
        return len(self.sequence_idxs)

    def __getitem__(self, index : int) -> Tuple[torch.Tensor]:
        """
        Convert a row of data from the Pandas dataframe into PyTorch tensor and return.
        
        Parameters
        ----------
            index : int
                Row (a datasample) of the dataframe to be processed.
        
        Returns
        -------
            (torch.Tensor, torch.Tensor) : Tuple of tensors where the first one represents input and second one the target.
        """
        # Reindex
        index_site_id, _, index_series_id, _, index_start_date, index_end_date = self.sequence_idxs[index]
        
        # Get spatial data frame in between start and end index
        df_tree = self.treenet_data_frame.get_tree_datetime_sequence_data_frame(index_site_id, index_series_id, index_start_date, index_end_date)
        df_tree_meta = self.treenet_data_frame.get_tree_metadata(index_series_id)

        inps = torch.tensor(df_tree.loc[:, self.input_feats].apply(pd.to_numeric).to_numpy(), dtype=torch.float32)
        inps = self.normalize_data(inps, self.normalization[0], self.norm_inputs)

        exog = torch.tensor(df_tree_meta.loc[self.exog_feats].apply(pd.to_numeric).to_numpy(), dtype=torch.float32)
        exog = self.normalize_data(exog, self.normalization[1], self.norm_exog)

        tgts = torch.tensor(df_tree.loc[:, self.target_feats].apply(pd.to_numeric).to_numpy(), dtype=torch.float32)
        tgts = self.normalize_data(tgts, self.normalization[2], self.norm_targets)

        if self.return_timestamp:
            tstamp = torch.tensor(index_end_date.value // 10 ** 9)
            return inps, exog, tgts, index_site_id, index_series_id, tstamp
        else:
            return inps, exog, tgts
