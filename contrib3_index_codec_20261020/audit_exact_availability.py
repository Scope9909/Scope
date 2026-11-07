"""Compare exact-history availability of frozen online state and selected codec."""
import argparse
import importlib.util
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'))
cli = parser.parse_args()
sys.argv = ['audit', cli.dataset]
spec = importlib.util.spec_from_file_location('online_codec', HERE / 'eval_online_srm_codec.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runner = module.runner
chosen = {'PEMS03': [22], 'PEMS04': [11], 'PEMS07': [117],
          'PEMS08': [7, 14, 19]}[cli.dataset]


def main():
    experiment, _, _, source, _ = runner.inputs()
    candidate = (HERE / 'packages' / 'online' /
                 f"{cli.dataset.lower()}_s1_test_fixed_{'_'.join(map(str, chosen))}_idx.npz")
    with (experiment / 'config.pkl').open('rb') as stream:
        args = pickle.load(stream)
    args.device = 'cpu'
    args.batch_size = 32
    args.cache_data_path = 'datasets'
    args.prob_factor = getattr(args, 'prob_factor', 5)
    if isinstance(getattr(args, 'k_hops', '1,2'), str):
        args.k_hops = [int(value) for value in args.k_hops.split(',')]
    args.pearson_adj = None
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = 'none'
    args.kd_alpha = 0
    loaders, _, scaler, _ = runner.datasets_utils.load_dataset(
        args, logging.getLogger('exact_availability_audit'))
    original = runner.OnlineResidualMemory.from_npz(source)
    coded = module.load_memory(candidate)
    index_path = (runner.ROOT / 'datasets' / cli.dataset /
                  f'{cli.dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz')
    with np.load(index_path, allow_pickle=False) as archive:
        query_count = len(archive['idx_test'])
    loader = loaders['test_loader']
    old = (loader.bs, loader.drop_last, loader.num_batch)
    loader.bs = 32
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / 32))
    changed = 0
    original_hits = 0
    coded_hits = 0
    queries = 0
    try:
        for loader_input, _ in loader.get_iterator():
            if queries >= query_count:
                break
            x, x_time, *extra = runner.unpack_dataloader_input(loader_input)
            take = min(len(x), query_count - queries)
            history = np.rint(scaler.inverse_transform(x[:take])[..., 0][
                :, :, original.node_ids]).astype(np.float32)
            tod = np.asarray(x_time[:take, -1, 0], dtype=np.int16)
            _, _, base_avail, _ = original.query(history, tod)
            _, _, coded_avail, _ = coded.query(history, tod)
            changed += int(np.count_nonzero(base_avail != coded_avail))
            original_hits += int(base_avail.sum())
            coded_hits += int(coded_avail.sum())
            queries += take
    finally:
        loader.bs, loader.drop_last, loader.num_batch = old
    result = dict(dataset=cli.dataset, chosen_nodes=chosen, queries=queries,
                  exact_availability_changed=changed,
                  exact_hits_original=original_hits, exact_hits_coded=coded_hits,
                  exact_availability_total=queries * original.predict_length * len(original.node_ids))
    target = HERE / 'packages' / 'online' / f'{cli.dataset.lower()}_exact_availability.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
