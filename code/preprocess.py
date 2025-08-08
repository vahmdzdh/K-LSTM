import argparse, pandas as pd, numpy as np, pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

def make_targets(df):
    # days_to_failure = max(days) - days  (train side)
    df = df.copy()
    df['days_to_failure'] = df.groupby('vehicleId')['days'].transform('max') - df['days']
    df['target'] = (df['days_to_failure'] <= 30).astype(int)
    return df

def make_targets_test(test_df, truth_df):
    # for test: max_day = last day in test + RUL (per vehicle)
    test_df = test_df.copy()
    t = truth_df.set_index('vehicleId')
    max_day = test_df.groupby('vehicleId')['days'].max()
    t['maximum_day'] = max_day + t['RUL']
    test_df['days_to_failure'] = t.loc[test_df['vehicleId'], 'maximum_day'].values - test_df['days']
    test_df['target'] = (test_df['days_to_failure'] <= 30).astype(int)
    return test_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--test_truth', required=True)
    ap.add_argument('--out', required=True, help='output pickle path')
    ap.add_argument('--seq_len', type=int, default=30)
    args = ap.parse_args()

    train = pd.read_csv(args.train, sep='\t')
    test = pd.read_csv(args.test, sep='\t')
    test_truth = pd.read_csv(args.test_truth, sep='\t')

    train = make_targets(train)
    test = make_targets_test(test, test_truth)

    sensor_cols = [c for c in train.columns if c.startswith('s')]

    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()

    train[sensor_cols] = imputer.fit_transform(train[sensor_cols])
    test[sensor_cols]  = imputer.transform(test[sensor_cols])

    train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
    test[sensor_cols]  = scaler.transform(test[sensor_cols])

    X_train = train[sensor_cols].to_numpy()
    y_train = train['target'].to_numpy()
    X_test  = test[sensor_cols].to_numpy()
    y_test  = test['target'].to_numpy()

    with open(args.out, 'wb') as f:
        pickle.dump({
            'sensor_cols': sensor_cols,
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'seq_len': args.seq_len
        }, f)
    print('Saved:', args.out)

if __name__ == '__main__':
    main()
