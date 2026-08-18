import pandas as pd
import numpy as np

from typing import Optional

def get_sequence_indices_treenet_tree(data: pd.DataFrame, window_size: int, max_dist: int = 48, index_filter: np.array = None, target_only_growth : Optional[int] = None) -> list:
    """
    Produce all the start and end index positions that are needed to produce the sub-sequences.

    Parameters
    ----------
        data : pd.DataFrame
            Sequence dataset from which to create sliced sub-sequences.

        window_size : int
            The desired length of each sub-sequence. Should be (input_sequence_length + target_sequence_length).
            E.g. if you want the model to consider past 100 time steps in order to predict the future 50 time
            steps, window_size = 100 + 50 = 150

        max_dist : int
            Maximum distance between start and end of the sequence in timesteps. Defaults to 48 (1 day).

        index_filter : np.array
            Array of indices to be considered as valid end sequence indices.

        target_only_growth : Optional[int]
            Return only series which have target indices within the growth period.

    Returns
    -------
        A list of tuples. Each tuple is (start_idx, end_idx) of a sub-sequence.
        The tuples should be used to slice the dataset into sub-sequences. These
        sub-sequences should then be passed into a function that slices them into input
        and target sequences.
    """

    index_data = data.copy()
    index_data['first_index'] = index_data.index - window_size + 1
    index_data['first_index_ts'] = index_data['ts'].shift(window_size - 1)
    index_data['first_index_site_id'] = index_data['site_id'].shift(window_size - 1)
    index_data['first_index_series_id'] = index_data['series_id'].shift(window_size - 1)
    timedelta = index_data['ts'] - index_data['first_index_ts']
    index_data['timesteps'] = timedelta.dt.components.days.abs() * 24 + timedelta.dt.components.hours.abs()
    index_data['index_filter_ok'] = (index_data.index).isin(index_filter)
    index_data['site_id_ok'] = index_data['site_id'] == index_data['first_index_site_id']
    index_data['series_id_ok'] = index_data['series_id'] == index_data['first_index_series_id']
    index_data['dist_ok'] = index_data['timesteps'] <= max_dist
    if target_only_growth is not None:
        index_data['target_only_growth'] = index_data['growth_period'].rolling(window_size).sum() >= target_only_growth
        index_data = index_data[index_data['index_filter_ok'] & index_data['site_id_ok'] & index_data['series_id_ok'] & index_data['dist_ok'] & index_data['growth_period'] & index_data['target_only_growth']]
    else:
        index_data = index_data[index_data['index_filter_ok'] & index_data['site_id_ok'] & index_data['series_id_ok'] & index_data['dist_ok']]
    index_data = index_data.reset_index()

    return [tuple(r) for r in index_data[['first_index_site_id', 'site_id', 'first_index_series_id', 'series_id', 'first_index_ts', 'ts']].to_numpy()]