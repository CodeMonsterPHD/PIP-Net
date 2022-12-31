import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import os

from lib.utils.utils import eval_all_metrics, SUBSET2category_idS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# organize the evaluation results in a table
def get_allresults_df(score_save) -> pd.DataFrame:
    score = 0
    data_dict = defaultdict(list)

    for model_type in ["image"]:
        data_dict["model"].append(model_type)
        val_f1s = []  # 5 x 28
        all_f1s = defaultdict(list)

        d_dict = torch.load(score_save)
        f1_dict = eval_all_metrics(
            d_dict["val_scores"], d_dict["test_scores"],
            d_dict["val_targets"], d_dict["test_targets"]
        )
        # todo: check
        for k, v in f1_dict.items():
            if isinstance(v, float):
                all_f1s[k].append(v * 100)
            else:
                all_f1s[k].append(np.array(v)[np.newaxis, :] * 100)

        val_f1s = np.vstack(all_f1s["val_none"])
        for e_type, c_category_ids in SUBSET2category_idS.items():
            e_f1s = np.mean(np.hstack([val_f1s[:, c:c+1] for c in c_category_ids]), 1)
            data_dict[f"val-{e_type}"].append("{:.2f} +- {:.2f}".format(
                np.mean(e_f1s), np.std(e_f1s)
            ))

        for k, values in all_f1s.items():
            if not k.endswith("none"):
                data_dict[k].append("{:.2f} +- {:.2f}".format(
                    np.mean(values), np.std(values)
                ))
                if "val" in k:
                    score = score+values[0]

    df = pd.DataFrame(data_dict)
    df.to_excel(score_save[:-4]+'.xlsx', index=False)
    return score/3, df
