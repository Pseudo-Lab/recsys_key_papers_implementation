import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd


def plot_rs_wp(model_prefix, title, metric='hit', version='standard'):
    '''
    Plots graphs of weighted pooling comparison
    '''
    df = pd.read_csv("model_scores_rs_{}.csv".format(version))

    plt.xlabel('K')
    plt.ylabel(metric + '@K')
    plt.title(title)

    for wp in [1, 10, .1, .01]:
        model = model_prefix + str(wp) + ".pt"
        cur_results = df.loc[df['model'] == model]

        x_vals = range(1, 16)
        y_vals = cur_results[metric].tolist()
        plt.plot(x_vals, y_vals, label='γ=' + str(wp), linewidth=1, marker='o', markersize=3)

    plt.axes().yaxis.set_minor_locator(AutoMinorLocator())
    plt.axes().set_xticks([1,3,5,7,9,11,13,15], minor=True)
    plt.legend()
    plt.grid()

    plt.show()
    #plt.savefig(title + '.png')
    #plt.close()

def plot_dense_wp(title, metric='hit'):
    '''
    Plots graphs of weighted pooling comparison
    '''
    df = pd.read_csv("model_scores_dense.csv")

    plt.xlabel('K')
    plt.ylabel(metric + '@K')
    plt.title(title)

    for wp in ['1', '10', '.1', '.01']:
        model = "dense_all_wp_" + wp + ".pt"
        cur_results = df.loc[df['model'] == model]

        x_vals = range(1, 16)
        y_vals = cur_results[metric].tolist()
        plt.plot(x_vals, y_vals, label='γ=' + wp, linewidth=1, marker='o', markersize=3)

    plt.axes().yaxis.set_minor_locator(AutoMinorLocator())
    plt.axes().set_xticks([1,3,5,7,9,11,13,15], minor=True)
    plt.legend()
    plt.grid()

    #plt.show()
    plt.savefig(title + '.png')
    plt.close()


def plot_versions(title, metric='hit'):
    plt.xlabel('K')
    plt.ylabel(metric + '@K')
    plt.title(title)
    x_vals = range(1, 16)

    df = pd.read_csv("model_scores_rs_standard.csv")

    model = "rs_all_wp_1.pt"
    cur_results = df.loc[df['model'] == model]
    y_vals = cur_results[metric].tolist()
    plt.plot(x_vals, y_vals, label='KPRN', linewidth=1, marker='o', markersize=3)

    df = pd.read_csv("model_scores_rs_sample.csv")

    model = "rs_all_sample_5_wp_1.pt"
    cur_results = df.loc[df['model'] == model]
    y_vals = cur_results[metric].tolist()
    plt.plot(x_vals, y_vals, label='KPRN sample 5', linewidth=1, marker='o', markersize=3)

    # model = "rs_all_no_rel_sample_5_wp_1.pt"
    # cur_results = df.loc[df['model'] == model]
    # y_vals = cur_results[metric].tolist()
    # plt.plot(x_vals, y_vals, label='kprn-r sample', linewidth=1, marker='o', markersize=3)



    # model = "rs_all_no_rel_wp_1.pt"
    # cur_results = df.loc[df['model'] == model]
    # y_vals = cur_results[metric].tolist()
    # plt.plot(x_vals, y_vals, label='kprn-r', linewidth=1, marker='o', markersize=3)


    plt.axes().yaxis.set_minor_locator(AutoMinorLocator())
    plt.axes().set_xticks([1,3,5,7,9,11,13,15], minor=True)
    plt.legend()
    plt.grid()

    #plt.show()
    plt.savefig(title + '.png')
    plt.close()


def plot_dense_vs_rs(title, metric='hit'):
    plt.xlabel('K')
    plt.ylabel(metric + '@K')
    plt.title(title)
    x_vals = range(1, 16)

    df = pd.read_csv("model_scores_rs_standard.csv")

    model = "rs_all_wp_1.pt"
    cur_results = df.loc[df['model'] == model]
    y_vals = cur_results[metric].tolist()
    plt.plot(x_vals, y_vals, label='standard', linewidth=1, marker='o', markersize=3)

    df = pd.read_csv("model_scores_dense.csv")

    model = "dense_all_wp_1.pt"
    cur_results = df.loc[df['model'] == model]
    y_vals = cur_results[metric].tolist()
    plt.plot(x_vals, y_vals, label='dense', linewidth=1, marker='o', markersize=3)

    #plt.axes().yaxis.set_minor_locator(AutoMinorLocator())
    plt.yticks(np.arange(.2,.9, step=.1))
    plt.axes().set_xticks([1,3,5,7,9,11,13,15], minor=True)
    plt.legend()
    plt.grid()

    #plt.show()
    plt.savefig(title + '.png')
    plt.close()


def plot_single_graph(model_prefix, title, metric='hit', version='standard'):
    df = pd.read_csv("model_scores_rs_{}.csv".format(version))

    plt.xlabel('K')
    plt.ylabel(metric + '@K')
    plt.title(title)

    wp = 1
    model = model_prefix + str(wp) + ".pt"
    cur_results = df.loc[df['model'] == model]

    x_vals = range(1, 16)
    y_vals = cur_results[metric].tolist()
    plt.plot(x_vals, y_vals, label='implementation', linewidth=2, marker='o', markersize=4)

    plt.axes().yaxis.set_minor_locator(AutoMinorLocator())
    plt.axes().set_xticks([1,3,5,7,9,11,13,15], minor=True)
    plt.legend()
    plt.grid()

    #plt.show()
    plt.savefig(title + '.png')
    plt.close()

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_vs_paper(title, metric='hit'):

    labels = [metric + '@' + str(5), metric + '@' + str(10), metric + '@' + str(15)]
    our_vals = [.691, .806, .857]
    paper_vals = [.717, .823, .881]


    x = np.arange(len(labels))  # the label locations
    width = 0.28  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, our_vals, width, label='implementation')
    rects2 = ax.bar(x + width/2, paper_vals, width, label='paper')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(metric + '@K')
    ax.set_xlabel('K')
    ax.set_title(title)
    ax.set_yticks(np.arange(0,1, step=.1))
    ax.set_xticks(x)
    ax.set_xticklabels([5,10,15])

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    ax.legend()
    #plt.show()
    plt.savefig(title + '.png')
    plt.close()


def plot_paths_baseline(title, paths_vals, our_vals, metric='hit', model='model'):
    labels = [metric + '@' + str(5), metric + '@' + str(10), metric + '@' + str(15)]


    x = np.arange(len(labels))  # the label locations
    width = 0.28  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, our_vals, width, label=model)
    rects2 = ax.bar(x + width/2, paths_vals, width, label='# paths baseline')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(metric + '@K')
    ax.set_xlabel('K')
    ax.set_title(title)
    ax.set_yticks(np.arange(0,1, step=.1))
    ax.set_xticks(x)
    ax.set_xticklabels([5,10,15])

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    ax.legend()
    #plt.show()
    plt.savefig(title + '.png')
    plt.close()


def plot_mi_vs_kkbox(title, mi_res, kkbox_res, metric='hit'):
    labels = [metric + '@' + str(5), metric + '@' + str(10), metric + '@' + str(15)]

    x = np.arange(len(labels))  # the label locations
    width = 0.26  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, kkbox_res, width, label='KKBox (sparser)')
    rects2 = ax.bar(x + width/2, mi_res, width, label='MI (denser)')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(metric + '@K')
    ax.set_xlabel('K')
    ax.set_title(title)
    ax.set_yticks(np.arange(0,1, step=.1))
    ax.set_xticks(x)
    ax.set_xticklabels([5,10,15])

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    ax.legend()
    #plt.show()
    plt.savefig(title + '.png')
    plt.close()


def main():
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=14)    # fontsize of the x and y labels

    #plot_rs_wp("rs_all_sample_5_wp_", "rs kprn hit@K with Path Sampling", "hit", "sample")
    #plot_rs_wp("rs_all_sample_5_wp_", "rs kprn ndcg@K with Path Sampling", "ndcg", "sample")

    # plot_rs_wp("rs_all_no_rel_sample_5_wp_", "rs kprn-r hit@K with Path Sampling", "hit", "sample")
    # plot_rs_wp("rs_all_no_rel_sample_5_wp_", "rs kprn-r ndcg@K with Path Sampling", "ndcg", "sample")
    #
    # plot_rs_wp("rs_all_wp_", "rs kprn hit@K", "hit", "standard")
    # plot_rs_wp("rs_all_wp_", "rs kprn ndcg@K", "ndcg", "standard")
    #
    # plot_rs_wp("rs_all_no_rel_wp_", "rs kprn-r hit@K", "hit", "standard")
    # plot_rs_wp("rs_all_no_rel_wp_", "rs kprn-r ndcg@K", "ndcg", "standard")
    #
    # plot_versions("rs relation and sampling comparison hit@K", "hit")
    # plot_versions("rs relation and sampling comparison ndcg@K", "ndcg")

    #plot_dense_wp("dense kprn hit@K", "hit")
    #plot_dense_wp("dense kprn ndcg@K", "ndcg")

    #plot_dense_vs_rs("rs vs dense comparison ndcg@K (γ=1)", "ndcg")

    ###Presentation graphs###
    # plot_versions("KPRN vs KPRN Sample 5", "hit")
    # plot_single_graph("rs_all_wp_","Our Implementation", "hit")
    # plot_vs_paper("Our Implementation vs Paper's Result", "hit")
    # plot_dense_vs_rs("Standard Subnetwork vs Dense Subnetwork", "hit")
    #
    # plot_paths_baseline("KPRN vs # of Paths Baseline",[.683, .802, .856],[.691, .806, .857], model='KPRN')
    # plot_paths_baseline("KPRN vs # of Paths Baseline (Sample 5)", [.161, .322, .487], [.677, .797, .850],model='KPRN')

    plot_mi_vs_kkbox("Paper's KKBox vs MI Results", [0.676, 0.773, 0.832], [0.717, 0.823, 0.881])




if __name__ == "__main__":
    main()
