""" Build the user-agnostic global trajectory flow map from the sequence data """
import os
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import collections

preversed_times=5
def static(df):
    df=df.copy()
    df['datetime'] = pd.to_datetime(df['local_time'])
    print(f"整个数据集的时间跨度为：{df['datetime'].min()} to {df['datetime'].max()} ")
    poi_num=len(set(df['POI_id'].to_list()))
    cat_num=len(set(df['POI_catid'].to_list()))
    user_num=len(set(df['user_id'].to_list()))
    grouped = df.groupby('user_id')
    average_length = grouped.size().mean()
    print(f"poi_num: {poi_num}, cat_num: {cat_num}, user_num: {user_num}, average seq len: {average_length}")
def NYC_IN_GETNext():
    print(
        'now preprocess the dataset NYC used in the paper GETNext\n'
        'you must make sure that the NYC_train.csv... file have been in /dataset/NYC/\n'
        'Some of the POIs in this dataset correspond to different classes in different '
        'check-in records, and the function selects the class that appears most often as the class of the POI,\n'
        'In addition, we observed that the data concentrated in the two time zones -240 and -300 accounted for '
        'the majority of the data, so we discarded the data in other time zones')
    train_df = pd.read_csv('./dataset/NYC/NYC_train.csv')
    val_df = pd.read_csv('./dataset/NYC/NYC_val.csv')
    test_df = pd.read_csv('./dataset/NYC/NYC_test.csv')
    df = pd.concat([train_df, val_df, test_df])

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
    main_timezones = df['timezone'].value_counts().head(2).index
    df = df[df['timezone'].isin(main_timezones)]
    static(df)
    df.to_csv('./dataset/NYC/NYC.csv', index=False)
    return


def tsmc(dataset):
    print('now preprocess the dataset launched by Yang DingQi called tsmc2014\n',
          'you must make sure that the dataset_TSMC2014_NYC.txt... file have been in /dataset/dataset_tsmc2014/\n')
    if dataset==21:
        file_name="./dataset/dataset_tsmc2014/dataset_TSMC2014_NYC.txt"
        save_name="./dataset/dataset_tsmc2014/NYC.csv"
        preserved_timezone=2
    else:
        file_name = "./dataset/dataset_tsmc2014/dataset_TSMC2014_TKY.txt"
        save_name = "./dataset/dataset_tsmc2014/TKY.csv"
        preserved_timezone = 1
    # 定义列名列表
    col_names = ["user_id", "POI_id", "POI_catid", "poi_cat_name", "latitude", "longitude", "timezone", "local_time"]

    # 读取csv文件，并添加列名
    df = pd.read_csv(file_name, names=col_names, sep='\t')
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
    static(df)
    df.to_csv(save_name, index=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='dataset preprocess, this func make sure that the df have col '
                    'of user_id,POI_id,POI_catid,local_time')
    parser.add_argument('-dataset', type=int, required=True,
                        help='choose dataset 1 NYC_IN_GETNext 21 tsmc_NYC 22 tsmc_TKY')

    args = parser.parse_args()
    if args.dataset == 1:
        NYC_IN_GETNext()
    elif args.dataset == 21 or args.dataset == 22:
        tsmc(args.dataset)
