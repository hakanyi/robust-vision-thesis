#!/usr/bin/env python3

'''

Obtained from https://github.com/CNCLgithub/rooms-psiturk/tree/master
Scrapes data from database, then parses it to anonymize and make good for analysis

'''

from __future__ import division, print_function
import os
import os.path as osp
import json
import sys
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table


def parse_arguments():
    parser = argparse.ArgumentParser(description = "Parses participants.db",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--db_dir", type=str, required=True,
                        help = 'Directory to find participants.db in.')
    parser.add_argument("--table_name", type=str, required=True,
                        help = 'Table name')
    parser.add_argument("--out_dir", type = str, required=True,
                        help = 'Directory to save output csv to.')
    parser.add_argument("--exp_version", type = str, nargs ='+', default = ["0.1"],
                        help = 'Experiment version flag')
    parser.add_argument("--mode", type = str, default = "debug",
                        choices = ['debug', 'sandbox', 'live'],
                        help = 'Experiment mode')
    parser.add_argument("--trialsbyp", type = int, default = 128,
                        help = 'Number of trials expected per subject')
    parser.add_argument("--trialdata", type = str,
                        default = 'parsed_trials.csv',
                        help = 'Filename to dump parsed trial data')
    parser.add_argument("--questiondata", type = str,
                        default = 'parsed_questions.csv',
                        help = 'Filename to dump parsed trial data')
    args = parser.parse_args()
    return args

# Mostly from http://psiturk.readthedocs.io/en/latest/retrieving.html
def read_db(db_dir, table_name, codeversions, mode):
    # check if database exists
    db_pth = osp.join(db_dir, "participants.db")
    if not osp.exists(db_pth):
        raise Exception("could not find a participants.db file")
    data_column_name = "datastring"
    engine = create_engine("sqlite:///" + db_pth)
    metadata = MetaData()
    metadata.bind = engine
    table = Table(table_name, metadata, autoload=True)
    s = table.select()
    rows = s.execute()

    rawdata = []
    statuses =  [3, 4, 5, 7]
    for row in rows:
        if (row['status'] in statuses and
            row['mode'] == mode and
            row['codeversion'] in codeversions):
            str_data = row[data_column_name]
            proc_data = json.loads(str_data)
            rawdata.append(proc_data)

    conddict = {}
    ids = {}
    for part in rawdata:
        uniqueid = part['workerId'] + ':' + part['assignmentId']
        ids[uniqueid] = part['data'][0]['trialdata']['prolific_id']
        conddict[uniqueid] = part['condition']
    data = [part['data'] for part in rawdata]

    for part in data:
        for record in part:
            record['trialdata']['uniqueid'] = record['uniqueid']
            record['trialdata']['condition'] = conddict[record['uniqueid']]
            record['trialdata']['prolific_id'] = ids[record['uniqueid']]

    trialdata = pd.DataFrame([record['trialdata'] for part in data for
                              record in part if
                              ('IsInstruction' in record['trialdata'] and
                               not record['trialdata']['IsInstruction'])])

    qdat = []
    for part in rawdata:
        thispart = part['questiondata']
        thispart['uniqueid'] = part['workerId'] + ':' + part['assignmentId']
        qdat.append(thispart)
    questiondata = pd.DataFrame(qdat)

    return trialdata, questiondata

def parse_row(tname):
    # scene data
    tpath, _ = osp.splitext(tname)
    splits = tpath.split('_')
    idx, trial_type = splits
    new_row = {
        'idx' : idx,
        'type' : trial_type,
        'same_gt' : trial_type == "same-image",
        }
    return new_row

def read_add_info(db_dir, key):
    add_info = None
    add_info_pth = osp.join(db_dir, "additional_info.json")
    if osp.exists(add_info_pth):
        with open(add_info_pth) as f:
            add_info = json.load(f)
            if key in add_info:
                add_info = add_info[key]
            else:
                add_info = None
    return add_info

def read_excluded_subj(db_dir):
    pth = osp.join(db_dir, "excluded_workers.txt")
    excluded = []
    if osp.exists(pth):
        with open(pth) as f:
            for line in f:
                excluded.append(line.rstrip())
    return excluded

def main():

    args = parse_arguments()

    trs, qs = read_db(args.db_dir, args.table_name,
                      args.exp_version, args.mode)

    # check if additional info exist, load if so
    add_info = read_add_info(args.db_dir, args.table_name)

    # load excluded workers
    excluded = read_excluded_subj(args.db_dir)

    cl_qs = qs.rename(index=str, columns={'uniqueid': 'WID'})

    trs = trs.dropna()
    trs = trs.rename(index=str,
                     columns={'ReactionTime':'RT',
                              'uniqueid':'WID'})
    # row_data = pd.concat(trs.apply(parse_row, axis=1).tolist())
    # trs = trs[['TrialName', 'WID', 'RT', 'Response', 'condition', 'TrialOrder']]
    # trs = trs.merge(row_data, on = ['TrialName', 'WID'])

    trs = trs.merge(trs.TrialName.apply(lambda s: pd.Series(parse_row(s))),
                    left_index=True, right_index=True)
    trs["same_response"] = trs.Response.replace({"f": False, "j": True})

    # Make sure we have required responses per participant
    trialsbyp = trs.groupby('WID').aggregate({"TrialOrder" : lambda x : max(x) + 1,
                                              "prolific_id": lambda x : x.unique()})
    print(trialsbyp)
    good_wids = trialsbyp[trialsbyp.TrialOrder  == args.trialsbyp].index
    trs = trs[trs.WID.isin(good_wids)]
    # delete excluded workers
    trs = trs[~trs.prolific_id.isin(excluded)]

    # Assign random identifiers to each participant
    wid_translate = {}
    for i, wid in enumerate(good_wids):
        wid_translate[wid] = i

    trs["ID"] = trs.WID.apply(lambda x: wid_translate[x])
    if add_info is not None:
        for key, value in add_info.items():
            trs[key] = value

    # randomize by deleting the worker ID
    trs = trs.drop(columns=['prolific_id'])

    os.makedirs(args.out_dir, exist_ok=True)
    out = osp.join(args.out_dir, args.trialdata)
    trs.to_csv(out, index=False)

    # cl_qs = cl_qs[cl_qs.WID.isin(good_wids)]
    # cl_qs["ID"] = cl_qs.WID.apply(lambda x: wid_translate[x])

    # out = osp.join(args.out_dir, args.questiondata)
    # cl_qs[["ID", "instructionloops", "comments"]].to_csv(out, index=False)

if __name__ == '__main__':
    main()
