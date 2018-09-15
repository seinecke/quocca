import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from glob import glob
from astropy.time import Time
from datetime import datetime, timedelta
from tqdm import tqdm

import click


def parse_filename(file):
    name = os.path.basename(file)
    try:
        cam_name, method_name, time_stamp = name[:-4].split('_')
    except:
        raise ValueError('Filename does not match pattern.')
    return Time(time_stamp)


def read_result_files(path, keys, time_start=None, time_end=None, ):
    print('Gathering file names.')
    files = np.array(glob(path))
    
    print('Parsing timestamps.')
    times = np.array([parse_filename(f).decimalyear
                      for f in files])
    t_sort = np.argsort(times)
    files = files[t_sort]
    times = times[t_sort]
    if type(time_start) is str:
        time_start = Time(time_start).decimalyear
    if type(time_end) is str:
        time_end = Time(time_end).decimalyear
    if type(time_start) == 'astropy.time.core.Time':
        time_start = time_start.decimalyear
    if type(time_end) == 'astropy.time.core.Time':
        time_end = time_end.decimalyear
    if time_start is None:
        time_start = np.min(times)
    if time_end is None:
        time_end = np.max(times)
    
    print('Extract additional attributes from {} to {}.'
          .format(time_start, time_end))
    selection = (times > time_start) & (times < time_end)
    files = files[selection]
    extract = {k: [] for k in keys}
    extract['time'] = times[selection]
    for f in tqdm(files, total=len(files)):
        df = pd.read_csv(f)
        for k in keys:
            try:
                extract[k].append(df[k].values)
            except:
                extract[k].append(-np.ones(len(df)))
    return extract


def calculate_quantiles(extract, keys, max_r=200, max_mag=4.0):
    r = np.array([np.sqrt((extract['x'][i] - 250) ** 2 + (extract['y'][i] - 326.3) ** 2)
                  for i in range(len(extract['x']))])
    mag = np.array([extract['v_mag'][i] for i in range(len(extract['x']))])
    quantiles = {}
    for k in keys:
        quantiles[k] = np.array([np.percentile(extract[k][i][(r[i] < max_r) & (mag[i] < max_mag)],
                                               [15,50,85])
                                 for i in range(len(extract[k]))])
    return quantiles


def plot_quantiles(time, quantiles, ylabels=None, colors=None):
    width = (np.max(time) - np.min(time)) * 800
    height =len(quantiles) * 3
    fig, ax = plt.subplots(len(quantiles), figsize=(width, height), sharex=True)
    time_start = Time(np.min(time), format='decimalyear').datetime
    time_end = Time(np.max(time), format='decimalyear').datetime
    print(time_start, time_end)
    first_day = datetime(day=time_start.day,
                         month=time_start.month,
                         year=time_start.year)
    last_day = datetime(day=time_end.day,
                        month=time_end.month,
                        year=time_end.year)
    print((last_day - first_day).days)
    days_inbetween_date = [first_day + timedelta(i)
                           for i in range((last_day - first_day).days + 1)]
    days_inbetween = [Time(first_day + timedelta(i)).decimalyear
                      for i in range((last_day - first_day).days + 1)]
    print(days_inbetween)
    for d in days_inbetween:
        for i in range(len(quantiles)):
            ax[i].axvline(d, color='k', linestyle='--', linewidth=1)
    if 'AxesSubplot' in str(type(ax)):
        ax = [ax]
    if colors is None:
        colors = ['C{}'.format(i) for i in range(len(quantiles))]
    for i, (k, vals) in enumerate(quantiles.items()):
        ax[i].plot(time, vals[:,1], color=colors[i])
        ax[i].fill_between(time, vals[:,0], vals[:,2], alpha=0.3,
                           facecolor=colors[i], edgecolor='none')
        ax[i].set_xlim([np.min(time), np.max(time)])
        ax[i].set_ylim([0.0, 1.0])
        if ylabels is not None:
            ax[i].set_ylabel(ylabels[i])
    ax[-1].set_xlabel('Time')
    ax[-1].set_xticks(days_inbetween)
    ax[-1].set_xticklabels(['{}/{:02}/{:02}'.format(d.year, d.month, d.day)
                            for d in days_inbetween_date])
    return ax

@click.command()
@click.option('-i', '--input', type=str)
@click.option('-m', '--month', type=int)
def parse_and_plot(input, month):
    if not os.path.exists('iceact_data_{:02}.csv'.format(month)):
        if month != 12:
            days = (datetime(2017,month + 1,1) - datetime(2017,month,1)).days
        else:
            days = 31
        extract = read_result_files(
            input,
            ['visibility', 'b_fit', 'id', 'v_mag', 'x', 'y'],
            Time(datetime(2017,month,1,0,0,0)).decimalyear,
            Time(datetime(2017,month,1,0,0,0) + timedelta(days=days)).decimalyear
        )
        quantiles = calculate_quantiles(extract, ['visibility', 'b_fit'])
        try:
            quantiles['visibility'] = 1.0 - quantiles['visibility']
        except:
            pass
        time = extract['time']
        data_dict = {}
        for k, val in quantiles.items():
            data_dict.update(**{'{}_15'.format(k): val[:,0],
                                '{}_50'.format(k): val[:,1],
                                '{}_85'.format(k): val[:,2]})
        data_dict.update(time=extract['time'])
        df = pd.DataFrame(data_dict).to_csv('iceact_data_{:02}.csv'
                                            .format(month))
    else:
        df = pd.read_csv('iceact_data_{:02}.csv'.format(month))
        unique_keys = np.unique(df.columns.str[:-3])
        quantiles = {}
        for k in unique_keys:
            try:
                quantiles[k] = np.column_stack((df['{}_15'.format(k)].values,
                                                df['{}_50'.format(k)].values,
                                                df['{}_85'.format(k)].values))
            except KeyError:
                pass
            time = df['time']
    plot_quantiles(time, quantiles,
                   ylabels=['Cloudiness', 'Brightness'],
                   colors=['#7ac143', '#114477'])
    plt.tight_layout()
    plt.savefig('plot_{:02}.pdf'.format(month))


if __name__ == '__main__':
    parse_and_plot()
