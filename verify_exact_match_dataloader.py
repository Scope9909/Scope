"""Smoke-test exact-memory cache loading, shuffling, batching, and padding."""

import logging

import numpy as np

from utils.datasets_utils import load_dataset
from verify_exact_match_model import CACHE, _load_args


def main():
    logger = logging.getLogger('verify-exact-loader')
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    args = _load_args(True)
    dataloader, num_nodes, _, num_test_samples = load_dataset(args, logger)
    train_loader = dataloader['train_loader']
    val_loader = dataloader['val_loader']
    test_loader = dataloader['test_loader']

    with np.load(CACHE, allow_pickle=False) as archive:
        idx_train = np.array(archive['idx_train'], copy=True)
        exact_train = np.array(archive['exact_retrieval_train'], copy=True)
        available_train = np.array(
            archive['exact_available_train'], copy=True
        )
        exact_val_first = np.array(
            archive['exact_retrieval_val'][:args.batch_size], copy=True
        )
        available_val_first = np.array(
            archive['exact_available_val'][:args.batch_size], copy=True
        )
        exact_test_last = np.array(
            archive['exact_retrieval_test'][-1], copy=True
        )
        available_test_last = np.array(
            archive['exact_available_test'][-1], copy=True
        )

    val_input, _ = next(val_loader.get_iterator())
    if len(val_input) != 5:
        raise RuntimeError(f'Expected five model inputs, got {len(val_input)}')
    _, _, analog_val, exact_val, available_val = val_input
    val_exact_difference = float(np.max(np.abs(
        exact_val - exact_val_first
    )))
    val_available_equal = bool(np.array_equal(
        available_val, available_val_first
    ))

    np.random.seed(123)
    train_loader.shuffle()
    position_by_index = {
        int(index): position for position, index in enumerate(idx_train)
    }
    positions = np.asarray([
        position_by_index[int(index)] for index in train_loader.idx[:64]
    ])
    train_exact_difference = float(np.max(np.abs(
        train_loader.exact_retrieval_data[:64] - exact_train[positions]
    )))
    train_available_equal = bool(np.array_equal(
        train_loader.exact_available_data[:64], available_train[positions]
    ))

    padding = len(test_loader.idx) - num_test_samples
    if padding:
        test_exact_padding_difference = float(np.max(np.abs(
            test_loader.exact_retrieval_data[-padding:]
            - exact_test_last[None]
        )))
        test_available_padding_equal = bool(np.all(
            test_loader.exact_available_data[-padding:]
            == available_test_last[None]
        ))
    else:
        test_exact_padding_difference = 0.0
        test_available_padding_equal = True

    passed = (
        num_nodes == 307
        and analog_val.shape == exact_val.shape == available_val.shape
        and val_exact_difference == 0.0
        and val_available_equal
        and train_exact_difference == 0.0
        and train_available_equal
        and test_exact_padding_difference == 0.0
        and test_available_padding_equal
    )
    print(f'num_nodes={num_nodes}, num_test_samples={num_test_samples}')
    print(f'val_model_input_count={len(val_input)}')
    print(f'val_tensor_shape={exact_val.shape}')
    print(f'val_exact_max_abs_difference={val_exact_difference}')
    print(f'val_available_equal={val_available_equal}')
    print(f'train_shuffle_exact_max_abs_difference={train_exact_difference}')
    print(f'train_shuffle_available_equal={train_available_equal}')
    print(f'test_padding={padding}')
    print(
        f'test_padding_exact_max_abs_difference='
        f'{test_exact_padding_difference}'
    )
    print(f'test_padding_available_equal={test_available_padding_equal}')
    print(f'PASS={passed}')


if __name__ == '__main__':
    main()
