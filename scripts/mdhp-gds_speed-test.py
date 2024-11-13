import time
import torch
from ruamel.yaml import YAML
from easydict import EasyDict as edict
import logging
import regex as re
import pandas as pd

from mdhpnet.utils import initLoggers
from mdhpnet.utils.math import setRandomSeed
from mdhpnet.utils.mdhp_solver import solve_mdhp_params, gen_example_timestamps

# Output path for the results
DF_OUTPUT_PATH = "results/mdhp-gds_speed-test_gpu.csv"

def main(cfg):
    cfg = edict(cfg)
    basic_logger = logging.getLogger("BASIC")
    cuda_pattern = re.compile(r"cuda(:\d+)?")
    test_cfg_keys = list(cfg.TestConfigs[0].keys()) + ["window_cost", "throughput"]
    df = pd.DataFrame(columns=test_cfg_keys)

    for i, single_test_cfg in enumerate(cfg.TestConfigs):
        basic_logger.critical(f"Test {i} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        basic_logger.info(f"{single_test_cfg}")
        setRandomSeed(single_test_cfg.random_seed)
        elapsed_time = 0.0
        # Process each message window
        for idx in range(
            single_test_cfg.n_message_windows + single_test_cfg.n_warmup_windows
        ):
            timestamps = gen_example_timestamps(
                mdhp_dim=single_test_cfg.mdhp_dim,
                max_len=single_test_cfg.max_timestamp_len,
                min_len=single_test_cfg.min_timestamp_len,
            )
            n_messages = sum([len(ts) for ts in timestamps])
            start_time = time.time()
            _, _, _ = solve_mdhp_params(
                timestamps=timestamps,
                device=single_test_cfg.device,
                max_epochs=500,
                check_interval=25,
                use_torch_compile=single_test_cfg.use_torch_compile,
                optimize_likelihood=single_test_cfg.optimize_likelihood,
            )
            if cuda_pattern.match(single_test_cfg.device):
                torch.cuda.synchronize()
            end_time = time.time()
            if idx < single_test_cfg.n_warmup_windows:
                continue
            elapsed_time += end_time - start_time

        window_cost = elapsed_time / single_test_cfg.n_message_windows
        throughput = n_messages / elapsed_time
        basic_logger.info(f"Window Cost: {window_cost:.6f} seconds per window")
        basic_logger.info(f"Throughput: {throughput:.6f} messages per second")
        df.loc[i] = [single_test_cfg[k] for k in test_cfg_keys[:-2]] + [
            window_cost,
            throughput,
        ]

    df.to_csv(DF_OUTPUT_PATH, index=False)

if __name__ == "__main__":
    initLoggers()
    # Modify the config file for the tests
    cfg = YAML().load(open("configs/mdhp-gds_speed-test.yml", "r"))
    main(cfg)
