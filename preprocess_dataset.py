""" Build the user-agnostic global trajectory flow map from the sequence data """
import os
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import collections
import networkx as nx
import numpy as np

preversed_times=10
def static(df,user_shift_count,user_shift_total_time):
    df=df.copy()
    df['datetime'] = pd.to_datetime(df['local_time'])
    print(f"整个数据集的时间跨度为：{df['datetime'].min()} to {df['datetime'].max()} ")
    poi_num=len(set(df['POI_id'].to_list()))
    cat_num=len(set(df['POI_catid'].to_list()))
    user_num=len(set(df['user_id'].to_list()))
    grouped = df.groupby('user_id')
    average_length = grouped.size().mean()
    print(f"poi_num: {poi_num}, cat_num: {cat_num}, user_num: {user_num}, average seq len: {average_length}")



def tsmc(dataset):
    print('now preprocess the dataset launched by Yang DingQi called tsmc2014\n',
          'you must make sure that the dataset_TSMC2014_NYC.txt... file have been in /dataset/dataset_tsmc2014/\n')
    if dataset==11:
        file_name="./dataset/dataset_tsmc2014/dataset_TSMC2014_NYC.txt"
        save_name="./dataset/dataset_tsmc2014/NYC.csv"
        preserved_timezone=2
    elif dataset==12:
        file_name = "./dataset/dataset_tsmc2014/dataset_TSMC2014_TKY.txt"
        save_name = "./dataset/dataset_tsmc2014/TKY.csv"
        preserved_timezone = 1
    else:
        return
    # 定义列名列表
    col_names = ["user_id", "POI_id", "POI_catid", "poi_cat_name", "latitude", "longitude", "timezone", "local_time"]

    # 读取csv文件，并添加列名
    df = pd.read_csv(file_name, names=col_names, sep='\t',encoding='ISO-8859-1')
    main_timezones = df['timezone'].value_counts().head(preserved_timezone).index
    df = df[df['timezone'].isin(main_timezones)]
    lookup = collections.defaultdict(dict)
    for i, row in df.iterrows():
        if row['POI_catid'] not in lookup[row['POI_id']]:
            lookup[row['POI_id']][row['POI_catid']] = 1
        else:
            lookup[row['POI_id']][row['POI_catid']] += 1
    ament = {}
    for poiid, catids in lookup.items():
        if len(catids) > 1:
            max_key = max(catids, key=catids.get)
            ament[poiid] = max_key
    def update_catid(row):
        poiid = row['POI_id']
        if poiid in ament:
            return ament[poiid]
        else:
            return row['POI_catid']

    df['POI_catid'] = df.apply(update_catid, axis=1)
    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= preversed_times].index
    df = df[df["user_id"].isin(valid_users)]
    poi_counts = df.groupby("POI_id").size()
    valid_pois = poi_counts[poi_counts >= preversed_times].index
    df = df[df["POI_id"].isin(valid_pois)]

    df['date_time'] = pd.to_datetime(df['local_time'])

    # 创建一个新列来保存原始的时间戳
    df['original_timestamp'] = df['date_time']

    # 然后，对数据集按用户分组
    grouped = df.groupby('user_id')
    user_shift_count = {}
    user_shift_total_time = {}
    minutes = 15

    # 定义一个函数来处理每个用户的数据
    def process_user_data_shift(user_data):
        # 对时间戳进行排序
        user_data = user_data.sort_values('date_time')

        # 获取时间戳列
        timestamps = user_data['date_time']

        # 初始化移动时间和移动次数
        total_shift_time = pd.Timedelta(0)
        count = 0

        # 遍历时间戳
        for i in range(len(timestamps) - 1):
            # 获取当前时间戳和下一个时间戳
            current_timestamp = timestamps.iloc[i]
            next_timestamp = timestamps.iloc[i + 1]

            # 如果当前时间戳与下一个时间戳之间的间隔小于15分钟
            if next_timestamp - current_timestamp < pd.Timedelta(minutes=minutes):
                # 计算需要移动的时间
                shift_time = pd.Timedelta(minutes=minutes) - (next_timestamp - current_timestamp)
                if i == 0:
                    user_data.at[timestamps.index[i], 'date_time'] -= shift_time
                else:
                    up_shift = current_timestamp - timestamps.iloc[i - 1] - pd.Timedelta(minutes=minutes)
                    if up_shift >= shift_time:
                        user_data.at[timestamps.index[i], 'date_time'] -= shift_time
                    else:
                        user_data.at[timestamps.index[i], 'date_time'] -= up_shift
                        down_shift = shift_time - up_shift
                        user_data.at[timestamps.index[i + 1], 'date_time'] += down_shift

                total_shift_time += shift_time
                count += 1
        user_shift_total_time[user_data['user_id'].iloc[0]] = total_shift_time
        user_shift_count[user_data['user_id'].iloc[0]] = count
        # 获取当前用户的最小时间戳
        min_timestamp = user_data['date_time'].min()

        # 计算每个时间戳相对于最小时间戳的偏移量（以15分钟为单位）
        user_data['index'] = ((user_data['date_time'] - min_timestamp) / pd.Timedelta(minutes=15)).astype(int)
        return user_data

    # 对每个用户的数据应用处理函数
    shift_df = grouped.apply(process_user_data_shift)
    shift_df = shift_df.reset_index(drop=True)

    max_shift_count=50
    def keep_row(row):
        # 获取当前行的用户
        user = row['user_id']

        # 如果当前用户的移动次数大于 max_shift_count，则删除这一行
        if user_shift_count[user] > max_shift_count:
            return False
        else:
            return True

    # 应用 keep_row 函数来过滤数据集
    result_df = shift_df[shift_df.apply(keep_row, axis=1)]

    static(result_df,user_shift_count,user_shift_total_time)
    result_df = result_df.copy()
    result_df['hour_of_week'] = result_df['date_time'].dt.dayofweek * 24 + result_df['date_time'].dt.hour
    result_df['hour_of_day'] = result_df['date_time'].dt.hour
    result_df['day_of_week'] = result_df['date_time'].dt.dayofweek
    result_df.to_csv(save_name, index=False)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='dataset preprocess, this func make sure that the df have col '
                    'of user_id,POI_id,POI_catid,local_time')
    parser.add_argument('-dataset', type=int, required=True,
                        help='choose dataset 11 tsmc_NYC 12 tsmc_TKY')

    args = parser.parse_args()
    if args.dataset == 11 or args.dataset == 12:
        tsmc(args.dataset)
