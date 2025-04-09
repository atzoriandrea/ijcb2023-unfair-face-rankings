import gc  # Garbage collection for managing memory
import os.path  # File path operations
import time  # Timing operations
from copy import deepcopy  # Deep copy of objects
from config import *  # Configuration file with paths and model information
import operator  # Provides standard operators for key functions
import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Plotting and visualization
import math  # Mathematical functions
import seaborn as sns  # Enhanced statistical visualizations
import json  # Handling JSON files
from sklearn.metrics.pairwise import cosine_similarity  # Compute similarity between embeddings
from scipy.stats import ks_2samp  # Statistical test (Kolmogorov–Smirnov)
from statannotations.Annotator import Annotator  # Adds annotations to plots
import itertools  # Iteration utilities

# Disable pandas chained assignment warnings
pd.options.mode.chained_assignment = None

# Enable garbage collection explicitly
gc.enable()

# Flag to control accuracy computation (currently not used explicitly)
accuracies = False

# Function to identify underrepresented demographic groups based on exposure
# Used in fairness calculations
def disparate(data, row, reverse_mapping, k=10):
    indexes = list(range(10))
    exposure_vals = [1 / math.log(p+1) for p in range(1, k+1)]
    scores = list(row["scores"][:k])
    tp_sc = tp_scores(data, row["class"], row["matches"][:k], scores)
    fp_grps = fp_groups(data, row["class"], row["matches"][:k])
    exclude = [scores.index(tp) for tp in tp_sc]
    to_count = list(set(indexes).difference(exclude))
    row_res = {k: 0 for k, _ in reverse_mapping.items()}
    for i, fp in enumerate(fp_grps):
        row_res[fp] += exposure_vals[to_count[i]] / sum(exposure_vals)

    del row_res[row["group"]]
    max_exp = max(row_res.items(), key=operator.itemgetter(1))[0]
    is_underrepresented = []
    for key, value in row_res.items():
        if len(list(row_res)) == 1:
            is_underrepresented.append(key)
        elif value == 0 or value / row_res[max_exp] < 0.8:
            is_underrepresented.append(key)
    return list(set(is_underrepresented))


def get_idx_full(v):
    if v["genre"] == "Male":
        pred = [0, 2, 4]
    else:
        pred = [1, 3, 5]
    if v["ethnicity"] == "Asian":
        pred = pred[0]
    elif v["ethnicity"] == "Black":
        pred = pred[1]
    else:
        pred = pred[2]
    return pred


def get_idx_eth(v):
    if v["ethnicity"] == "Asian":
        pred = 0
    elif v["ethnicity"] == "Black":
        pred = 1
    else:
        pred = 2
    return pred


def get_idx_genre(v):
    if v["genre"] == "Male":
        pred = 0
    else:
        pred = 1
    return pred


def one_match(df, cls, matches):
    k_match = []
    for k in range(1, 11):
        found = False
        for m in matches[:k]:
            if df.loc[m]["class"] == cls:
                found = True
        k_match.append(found) #return False
    return k_match

def one_match_v2(matches, cls):
    if cls in matches["class"]:
        return True
    return False


def hit_rate(df, cls, matches):
    if not matches:
        return np.nan
    hits = 0
    for m in matches:
        if df.loc[m]["class"] == cls:
            hits += 1
    return hits / len(matches)


def fp_groups(df, cls, matches):
    fp = []
    for m in matches:
        if df.loc[m]["class"] != cls:
            fp.append(df.loc[m]["group"])
    return fp


def fp_attributes(df, cls, matches, features):
    fp = []
    for m in matches:
        if df.loc[m]["class"] != cls:
            f_vals = {}
            for f in features:
                f_vals[f] = df.loc[m][f]
            fp.append(f_vals)
    return fp


def tp_scores(df, cls, matches, scores):
    tp_scores = []
    for i, m in enumerate(matches):
        if df.loc[m]["class"] == cls:
            tp_scores.append(scores[i])
    return tp_scores


def scores_groups(df, cls, matches, scores, mapping):
    sc_grp = [[] for _ in range(len(mapping))]
    for i, m in enumerate(matches):
        if df.loc[m]["class"] != cls:
            sc_grp[mapping[df.loc[m]["group"]]].append(scores[i])
    return sc_grp


c_feat = ['age', 'smile', 'moustache', 'beard', 'sideburns', 'head_roll', 'head_yaw', 'head_pitch', 'blur', 'exposure',
          'noise']


def get_soft_features(path, feat, featfile):
    k = "/".join(path.strip().split(" ")[0].split("/")[-2:])
    v = []
    try:
        v = featfile[k][feat]
    except:
        v = np.nan
    return v


def get_gr_mappings(unique_grps):
    n_grps = len(unique_grps)
    if n_grps == 2:
        d = {0: "Man", 1: "Woman"}
        # r_d = dict(reversed(list(d.items())))
    elif n_grps == 3:
        d = {0: "Asian", 1: "Black", 2: "Caucasian"}
        # r_d = dict(reversed(list(d.items())))
    else:
        d = {0: 'AM', 1: 'AW', 2: 'BM', 3: 'BW', 4: 'CM', 5: 'CW'}
        # r_d = dict(reversed(list(d.items())))
    r_d = {v: k for k, v in d.items()}
    return d, r_d


def compute_ranking(df, embedding, k):
    cmps = df[df['probe'] == 0]
    indexes = np.asarray(cmps.index.tolist())
    Y = np.asarray(cmps["embeddings"].to_list())
    similarities = cosine_similarity(np.expand_dims(embedding, axis=0), Y)
    sorted_sim_indexes = np.argsort(similarities, axis=1)[0, -k:][::-1]
    sorted_indexes = indexes[sorted_sim_indexes]
    sorted_similarities = similarities[0, sorted_sim_indexes]
    return sorted_indexes, sorted_similarities


def generate_base_dataframe(mw, num, ds_info, mapping, k, gallery_mw=None):
    s = mw.groupby("class")["class"].count()
    s = s[s >= num]
    df_sel = mw[mw["class"].isin(s.index)].groupby("class").head(num)

    del mw
    gc.collect()

    df_sel["probe"] = 0
    df_sel["group"] = df_sel["path"].apply(lambda x: ds_info[x.split("/")[-2]])
    df_sel.loc[df_sel.groupby('class')['probe'].head(3 if num >= 10 else 2).index, 'probe'] = 1
    df_sel["embeddings"] = df_sel["embeddings"].apply(lambda x: np.nan if np.isnan(x).any() or np.isinf(x).any() else x)
    df_sel.dropna(axis=0, inplace=True)
    df_sel_ref = df_sel.loc[df_sel.query("probe == 1").index]

    if gallery_mw is None:
        df_sel_cmp_emb = df_sel.loc[df_sel.query("probe == 0").index].groupby(["group",'class'])[["embeddings"]].mean().reset_index()
    else:
        df_sel_gallery = gallery_mw[gallery_mw["class"].isin(s.index)].groupby("class").head(num)
        del gallery_mw
        gc.collect()
        df_sel_gallery["probe"] = 0
        df_sel_gallery["group"] = df_sel_gallery["path"].apply(lambda x: ds_info[x.split("/")[-2]])
        df_sel_gallery.loc[df_sel_gallery.groupby('class')['probe'].head(3 if num >= 10 else 2).index, 'probe'] = 1
        df_sel_gallery["embeddings"] = df_sel_gallery["embeddings"].apply(
            lambda x: np.nan if np.isnan(x).any() or np.isinf(x).any() else x)
        df_sel_gallery.dropna(axis=0, inplace=True)
        df_sel_cmp_emb = df_sel_gallery.loc[df_sel_gallery.query("probe == 0").index].groupby(["group", 'class'])[
            ["embeddings"]].mean().reset_index()

    # df_sel_cmp = pd.merge(df_sel_cmp_emb, df_sel_cmp_feat, how='inner')
    df_sel = pd.concat([df_sel_ref, df_sel_cmp_emb]).sort_values("class").replace(np.nan, 0).sort_values(["class", "probe"], ascending=[True, False])
    del df_sel_cmp_emb, df_sel_ref  # , df_sel_cmp_feat, df_sel_cmp
    gc.collect()
    df_sel = df_sel.groupby(["group", "class"]).head(df_sel.groupby(["group"]).count().min()[0])
    df_sel['group'].replace(mapping, inplace=True)
    df_sel.reset_index(inplace=True)

    ref_idx = np.asarray(df_sel[df_sel['probe'] == 1].index.tolist())
    cmps_idx = np.asarray(df_sel[df_sel['probe'] == 0].index.tolist())
    print("Computing similarities...")
    start_time = time.time()
    X = df_sel[df_sel['probe'] == 1]["embeddings"].to_list()
    Y = df_sel[df_sel['probe'] == 0]["embeddings"].to_list()
    similarities = cosine_similarity(X, Y)
    sorted_idxs = cmps_idx[np.argsort(similarities, axis=1)[:, -k:]]
    sorted_sims = np.sort(similarities, axis=1)[:, -k:]

    df_sel["matches"] = [[] for _ in range(len(df_sel))]
    df_sel["scores"] = [[] for _ in range(len(df_sel))]
    for i in range(len(sorted_idxs)):
        df_sel.loc[:, 'matches'].loc[ref_idx[i]] = sorted_idxs[i, ::-1].tolist()
        df_sel.loc[:, 'scores'].loc[ref_idx[i]] = sorted_sims[i, ::-1].tolist()
    del similarities, sorted_idxs, sorted_sims
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time))
    gc.collect()
    df_sel["class"] = df_sel["class"].astype(np.int64)
    df_sel["path"] = df_sel["path"].astype(str)
    df_sel["group"] = df_sel["group"].astype(str)
    df_sel["probe"] = df_sel["probe"].astype(bool)
    return df_sel


def get_exposure(data, reverse_mapping, k=None):
    indexes = list(range(10))
    exposure_vals = [1 / math.log2(p + 1) for p in range(1, 11)]
    sub = data
    if k is not None:
        sub = sub[sub["k"] == k]
    exposure = {}
    for index, row in sub.iterrows():
        scores = list(row["scores"])
        tp_sc = row["tp_scores"]
        fp_grps = row["fp_groups"]
        exclude = [scores.index(tp) for tp in tp_sc]
        to_count = list(set(indexes).difference(exclude))
        row_res = {k: 0 for k, _ in reverse_mapping.items()}
        for i, fp in enumerate(fp_grps):
            row_res[fp] += exposure_vals[to_count[i]] / sum(exposure_vals)
        exposure[index] = row_res
    return pd.DataFrame().from_dict(exposure, orient="index")


# %%
def get_visibility(data, reverse_mapping, k=None):
    # indexes = list(range(10))
    # exposure_vals = [1 / math.log2(p+1) for p in range(1,11)]
    sub = data
    if k is not None:
        sub = sub[sub["k"] == k]
    visibility = {}
    for index, row in sub.iterrows():
        fp_grps = row["fp_groups"]
        row_res = {k: 0 for k, _ in reverse_mapping.items()}  # {"AM" :0, "AW": 0, "BM":0, "BW":0, "CM":0, "CW":0}
        for i, fp in enumerate(fp_grps):
            row_res[fp] += 1 / 10
        visibility[index] = row_res
    return pd.DataFrame().from_dict(visibility, orient="index")



info = {
    "DiveFace": {
        "test_set_file": "/media/Workspace/Datasets/IJCB/DiveFaceResizedFaceX/testHR.txt",
        "json": "/media/Workspace/Datasets/IJCB/DiveFaceResizedFaceX/old_data/DiveFace_resized_update.json",
        "gr_fun": get_idx_full
    },
    "VGGFace2": {
        "test_set_file": "/media/Workspace/Datasets/IJCB/VGG-Face2/data/vggface2_test/testHR.txt",
        "json": "/media/Workspace/Datasets/IJCB/VGG-Face2/data/vggface2_test/test_HR.json",
        "gr_fun": get_idx_genre
    },
    "CelebA": {
        "test_set_file": "/media/Workspace/Datasets/JSTSP/FromIJCB/CelebA/testHR.txt",
        "json": "/media/Workspace/Datasets/JSTSP/FromIJCB/CelebA/test_HR.json",
        "gr_fun": get_idx_genre
    },
    "RFW": {
        "test_set_file": "/media/Workspace/Datasets/JSTSP/RFW/testHR.txt",
        "json": "/media/Workspace/Datasets/JSTSP/RFW/test_HR.json",
        "gr_fun": get_idx_eth
    },
    "BUPT": {
        "test_set_file": "/media/Workspace/Datasets/JSTSP/BUPT/testHR.txt",
        "json": "/media/Workspace/Datasets/JSTSP/BUPT/test_HR.json",
        "gr_fun": get_idx_eth
    },
}


models_data = models_data_HR_HR

def exec(gallery_dict=None, re_rank=False):
    basepath = "/media/Workspace/Thesis_EXT/"
    for dataset in list(info.keys()):
        print(dataset)
        models = models_data[dataset]
        if gallery_dict:
            gallery_models = gallery_dict[dataset]
        dataset_info = info[dataset]

        with open(dataset_info["test_set_file"], "r") as txtfile:
            test_set = txtfile.readlines()
        for i, r in enumerate(test_set):
            test_set[i] = r.strip().split(" ")[0]

        ds_info = {}
        with open(dataset_info["json"], "r") as jsonpointer:  # dataset_info["json"]
            jsonfile = json.load(jsonpointer)
        for key, value in jsonfile.items():
            folder = value["path"].split("/")[-2]
            if folder not in ds_info:
                ds_info[folder] = dataset_info["gr_fun"](value)

        model_wise = []
        for m in models:
            model_embedding = np.load(m).astype(np.float32)
            data = {}
            for i, r in enumerate(model_embedding):
                data[i] = {
                    "embeddings": r[:512],
                    "class": r[512].astype(np.int64),
                    "path": test_set[i]
                }
            df = pd.DataFrame.from_dict(data, orient="index")
            model_wise.append(df)

        gallery_mw = None
        if gallery_dict:
            gallery_mw = []
            for g_m in gallery_models:
                model_embedding_gallery = np.load(g_m).astype(np.float32)
                data_gallery = {}
                for i, r in enumerate(model_embedding_gallery):
                    data_gallery[i] = {
                        "embeddings": r[:512],
                        "class": r[512].astype(np.int64),
                        "path": test_set[i]
                    }
                df_gall = pd.DataFrame.from_dict(data_gallery, orient="index")
                gallery_mw.append(df_gall)

        rappr = model_wise[0]
        rappr["group"] = rappr["path"].apply(lambda x: ds_info[x.split("/")[-2]])

        mapping, reverse_mapping = get_gr_mappings(np.unique(rappr["group"]))

        min_num = 10 if dataset not in ["RFW"] else 5
        #""""""
        if not os.path.exists("/media/Workspace/Ranking/" + dataset + "_ds_repr_v2.png"):
            available = rappr.groupby('class').head(1).groupby("group").count().reset_index()[["group", "embeddings"]]
            s = rappr.groupby("class")["class"].count()
            s = s[s >= min_num]
            used = rappr[rappr["class"].isin(s.index)]
            used = used.groupby('class').head(1).groupby("group").count().reset_index()[["group", "embeddings"]]
            users = pd.merge(left=available, right=used, how="inner", on="group")
            users.rename(columns={"embeddings_x": "available", "embeddings_y": "used"}, inplace=True)
            users['group'].replace(mapping, inplace=True)

            fig = plt.figure(figsize=(10, 2))
            av = sns.barplot(data=users, x="group", y="available", color=sns.color_palette("pastel")[0], label="# ID(s)")
            ax = sns.barplot(data=users, x="group", y="used", color=sns.color_palette()[0], label="# ID(s) w/ "+str(min_num)+" img.")
            for p in ax.patches:
                av.annotate(format(int(p.get_height()), 'd'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 5),  # if p.get_height() < 200 else (0,-15),
                            textcoords='offset points')

            y_max = users["available"].max()

            ax.set_title(dataset)
            ax.axhline(users["used"].min(), color="orange", linestyle="--")  # sns.color_palette("husl", 9)[3])
            ax.set_ylim(0, y_max + (.2 * y_max))
            ax.set_ylabel("# Identities")
            plt.legend(bbox_to_anchor=(0.8, 1.24), loc="upper center", ncols=6)
            plt.savefig("/media/Workspace/Ranking - Preliminaries/Suspects_ranking/" + dataset + "_ds_repr_v2.png", dpi=400, bbox_inches="tight")

            del rappr, used, available
            gc.collect()
        num = min_num
        ks = [users["used"].min() * np.unique(users["group"]).shape[0]]
        model_ranking = []
        for i, mw in enumerate(model_wise):
            for k in ks:

                filepath = os.path.join(basepath, dataset, "_".join([model_names[i], str(min_num)])+"_full.pkl")
                if os.path.exists(filepath):
                    print("Skipping %s: file already exists" % filepath)
                    break
                gallery = gallery_mw[i] if gallery_dict else None
                df_sel_gen = generate_base_dataframe(mw, num, ds_info=ds_info, mapping=mapping, k=10, gallery_mw=gallery)

                df_sel = df_sel_gen
                folder_path = os.path.join(basepath, dataset, "Baseline")
                filepath = os.path.join(basepath, dataset, "Instance-Relative", "_".join([model_names[i], str(k), "baseline"]) + ".pkl")
                gc.collect()
                df_sel["one_match"] = df_sel[["class", "matches"]].apply(lambda x: one_match(df_sel, *x), axis=1)  # one_match_v2(df_sel.iloc[x.matches], x["class"]), axis=1)
                df_sel["tp_scores"] = df_sel[["class", "matches", "scores"]].swifter.apply(lambda x: tp_scores(df_sel, *x), axis=1)
                df_sel["fp_groups"] = df_sel[["class", "matches"]].swifter.apply(lambda x: fp_groups(df_sel, *x),axis=1)
                df_sel["k"] = min_num
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                if not os.path.exists(os.path.abspath(os.path.join(filepath, os.pardir))):
                    os.makedirs(os.path.abspath(os.path.join(filepath, os.pardir)))

                df_sel.to_pickle(filepath, protocol=4)
                del df_sel
                gc.collect()

                gc.collect()
                df_sel_gen["one_match"] = df_sel_gen[["class", "matches"]].apply(lambda x: one_match(df_sel_gen, *x), axis=1)
                df_sel_gen["tp_scores"] = df_sel_gen[["class", "matches", "scores"]].swifter.apply(lambda x: tp_scores(df_sel_gen, *x), axis=1)
                df_sel_gen["fp_groups"] = df_sel_gen[["class", "matches"]].swifter.apply(lambda x: fp_groups(df_sel_gen, *x),axis=1)
                df_sel_gen["k"] = min_num
                if not os.path.exists(os.path.join(basepath, dataset, "Baseline")):
                    os.makedirs(os.path.join(basepath, dataset, "Baseline"))
                df_sel_gen.to_pickle(os.path.join(basepath, dataset, "Baseline",
                                              "_".join([model_names[i], str(k)]) + ".pkl"),protocol=4)
                model_ranking.append(df_sel_gen)
                del df_sel_gen
                gc.collect()


        combinations = []
        for pair in itertools.combinations(list(mapping.values()), 2):
            combinations.append(pair)

        fig, ax = plt.subplots(2, 7, figsize=(25, 10))

        for i, r in enumerate(model_ranking):
            exposure = get_exposure(r, k=10, reverse_mapping=reverse_mapping)
            exposure[exposure == 0] = np.nan

            pvalues = [ks_2samp(exposure[x], exposure[y]).pvalue for x, y in combinations]
            formatted_pvalues = [f'p={pvalue:.2e}' if pvalue <= 0.05 else None for pvalue in pvalues]

            ax[0, i].set_title(model_names[i])
            ax[0, i].set_ylabel("Exposure")
            sns.boxplot(data=exposure, ax=ax[0, i])
            annotator = Annotator(ax[0, i], combinations, data=exposure)
            annotator.set_pvalues(pvalues)  # set_custom_annotations(formatted_pvalues)
            annotator.annotate()
        for i, r in enumerate(model_ranking):
            visibility = get_visibility(r, k=10, reverse_mapping=reverse_mapping)
            visibility[visibility == 0] = np.nan

            pvalues = [ks_2samp(visibility[x], visibility[y]).pvalue for x, y in combinations]

            ax[1, i].set_title(model_names[i])
            ax[1, i].set_ylabel("Visibility")
            sns.boxplot(data=visibility, ax=ax[1, i])

            annotator = Annotator(ax[1, i], combinations, data=visibility)
            annotator.set_pvalues(pvalues)  # set_custom_annotations(formatted_pvalues)
            annotator.annotate()
        plt.savefig("output/" + dataset + "_exp_vis_w_stat.png",
                    dpi=400, bbox_inches="tight")

        fig, ax = plt.subplots(2, 7, figsize=(25, 10))

        for i, r in enumerate(model_ranking):
            exposure = get_exposure(r, k=10, reverse_mapping=reverse_mapping)
            exposure[exposure == 0] = np.nan

            deltas = [abs(exposure[x].mean() - exposure[y].mean()) for x, y in combinations]
            formatted_deltas = [f'Δ={delta:.2e}' for delta in deltas]

            ax[0, i].set_title(model_names[i])
            ax[0, i].set_ylabel("Exposure")
            sns.boxplot(data=exposure, ax=ax[0, i])
            annotator = Annotator(ax[0, i], combinations, data=exposure)
            annotator.set_custom_annotations(formatted_deltas)
            annotator.annotate()

        for i, r in enumerate(model_ranking):
            visibility = get_visibility(r, k=10, reverse_mapping=reverse_mapping)
            visibility[visibility == 0] = np.nan

            deltas = [abs(visibility[x].mean() - visibility[y].mean()) for x, y in combinations]
            formatted_deltas = [f'Δ={delta:.2e}' for delta in deltas]

            ax[1, i].set_title(model_names[i])
            ax[1, i].set_ylabel("Visibility")
            sns.boxplot(data=visibility, ax=ax[1, i])

            annotator = Annotator(ax[1, i], combinations, data=visibility)
            annotator.set_custom_annotations(formatted_deltas)
            annotator.annotate()
        plt.savefig("output/" + dataset + "_deltas_exp_vis_.png",
                    dpi=400, bbox_inches="tight")

        all_models = pd.concat(model_ranking)

        fig, ax = plt.subplots(1, 1, figsize=(25, 10))

        plt.rcParams["axes.grid.axis"] = "y"
        plt.rcParams["axes.grid"] = True

        exposure = get_exposure(all_models, k=10, reverse_mapping=reverse_mapping)
        exposure[exposure == 0] = np.nan

        pvalues = [ks_2samp(exposure[x], exposure[y]).pvalue for x, y in combinations]
        deltas = [abs(exposure[x].mean() - exposure[y].mean()) for x, y in combinations]

        formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]
        formatted_deltas = [f'Δ={delta:.2e}' for delta in deltas]
        formatted_infos = [" - ".join([pv, delta]) for pv, delta in zip(formatted_pvalues, formatted_deltas)]

        sns.boxplot(data=exposure, ax=ax)
        ax.set_ylabel("Exposure")
        ax.set_yticks(list(np.arange(0., 1.1, 0.1)))
        ax.set_facecolor("white")

        annotator = Annotator(ax, combinations, data=exposure)
        annotator.set_custom_annotations(formatted_infos)
        annotator.annotate()
        # ax.grid(axis='y')
        plt.savefig("output/exposure_" + dataset + "_.png", dpi=400,
                    bbox_inches="tight")

        fig, ax = plt.subplots(1, 1, figsize=(25, 10))

        plt.rcParams["axes.grid.axis"] = "y"
        plt.rcParams["axes.grid"] = True

        visibility = get_visibility(all_models, k=10, reverse_mapping=reverse_mapping)
        visibility[visibility == 0] = np.nan

        pvalues = [ks_2samp(visibility[x], visibility[y]).pvalue for x, y in combinations]
        deltas = [abs(visibility[x].mean() - visibility[y].mean()) for x, y in combinations]

        formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]
        formatted_deltas = [f'Δ={delta:.2e}' for delta in deltas]
        formatted_infos = [" - ".join([pv, delta]) for pv, delta in zip(formatted_pvalues, formatted_deltas)]

        sns.boxplot(data=visibility, ax=ax)
        ax.set_ylabel("Visibility")
        ax.set_yticks(list(np.arange(0., 1.1, 0.1)))
        ax.set_facecolor("white")

        annotator = Annotator(ax, combinations, data=visibility)
        annotator.set_custom_annotations(formatted_infos)
        annotator.annotate()
        # ax.grid(axis='y')
        plt.savefig("output/visibility_" + dataset + "_.png",
                    dpi=400, bbox_inches="tight")


if __name__ == '__main__':
    exec(gallery_dict=None, re_rank=False)
