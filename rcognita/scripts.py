import distutils.errors
import os
import datetime
from distutils.dir_util import copy_tree, remove_tree

def aggregate_runs(from_=None, till=None, recent=None):
    assert recent or from_, "Please specify runs to aggregate"
    assert not recent or not from_
    initial_cwd = os.getcwd()
    os.chdir('multirun')
    by_date = sorted(os.listdir('.'))
    runs = []
    if isinstance(recent, datetime.timedelta):
        till = datetime.datetime.now()
        from_ = till - recent
        recent = None
    if from_:
        if not till:
            till = datetime.datetime.now()
        from_idx = None
        while not from_idx:
            try:
                from_idx = by_date.index(str(from_.date()))
            except:
                from_idx = None
                from_ += datetime.timedelta(days=1)
            if from_ > till:
                os.chdir(initial_cwd)
                return []
        till_idx = None
        while not till_idx:
            try:
                till_idx = by_date.index(str(till.date()))
            except:
                till_idx = None
                till -= datetime.timedelta(days=1)
            if from_ > till:
                os.chdir(initial_cwd)
                return []
        by_date = by_date[from_idx: till_idx + 1]
        runs = []
        for date_dir in by_date:
            os.chdir(date_dir)
            appended_runs = list(map(lambda s: date_dir + "/" + s, sorted(os.listdir('.'))))
            for run in appended_runs:
                if from_ <= datetime.datetime.strptime(run, "%Y-%m-%d/%H-%M-%S") <= till:
                    runs.append(run)
            os.chdir('..')
    elif recent:
        while recent != 0:
            last = by_date.pop(-1)
            os.chdir(last)
            appended_runs = sorted(os.listdir('.'))[-recent:]
            recent -= len(appended_runs)
            runs += list(map(lambda s: last + "/" + s, appended_runs))
            os.chdir('..')
    os.chdir(initial_cwd)
    return runs

def merge_runs(from_=None, till=None, recent=None, name=None, symlink=False):
    runs = aggregate_runs(from_=from_, till=till, recent=recent)
    if name is None:
        if from_:
            if not till:
                till = datetime.datetime.now()
            name = f"merged runs {str(from_).split('.')[0]}-{str(till).split('.')[0]}"
        elif recent:
            name = f"merged runs {str(datetime.datetime.now()).split('.')[0]} recent {recent}"
    os.mkdir(name)
    path = os.path.abspath(name)
    if not runs:
        return
    os.chdir('multirun')
    subrun_ids = list(filter(lambda x: x.isnumeric(), os.listdir(runs[0])))
    subrun_paths = {}
    for subrun_id in subrun_ids:
        subrun_paths[subrun_id] = path + '/' + subrun_id
        os.mkdir(subrun_paths[subrun_id])
    for run in runs:
        os.chdir(run)
        for subrun_id in subrun_ids:
            try:
                if symlink:
                    src = os.path.abspath(subrun_id)
                    if not os.path.isdir(src):
                        raise RuntimeError()
                    os.symlink(src, subrun_paths[subrun_id] + '/' + run.replace('/', '_'))
                else:
                    copy_tree(subrun_id, subrun_paths[subrun_id] + '/' + run.replace('/', '_'))
            except (distutils.errors.DistutilsError, RuntimeError) as e:
                os.chdir('../../..')
                remove_tree(path)
                raise RuntimeError(f"Failed to merge runs. Run {run} is not compatible with {runs[0]}. Perhaps select "
                                   f"runs are not a part of the same experiment.")
        os.chdir('../..')
    os.chdir('..')






