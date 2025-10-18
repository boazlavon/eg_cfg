# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

def build_dataframe(benchmark: str) -> pd.DataFrame:
    """Return dataframe for the given benchmark."""
    if benchmark == "mbpp":
        df = pd.DataFrame(data=[], columns=["Model", "Method", "MBPP", "MBPP-ET"])
        df.loc[len(df)] = ['GPT-4', 'Baseline LLM', 68.3, 49.2]
        df.loc[len(df)] = ['GPT-4', 'Self-Collaboration [44]', 78.9, 62.1]
        df.loc[len(df)] = ['GPT-4', 'Self-Debugging [11]', 80.6, pd.NA]
        df.loc[len(df)] = ['GPT-4', 'MetaGPT [26]', 87.7, pd.NA]
        df.loc[len(df)] = ['GPT-4', 'MapCoder [14]', 83.1, 57.5]
        df.loc[len(df)] = ['GPT-4o', 'LPW [27]', 84.8, 65.8]
        df.loc[len(df)] = ['Claude-Sonnet-3.5', 'Baseline LM [12]', 88.7, pd.NA]
        df.loc[len(df)] = ['Claude-Sonnet-3.5', 'QualityFlow [12]', 94.2, pd.NA]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'Baseline LLM', 82.8, 64.8]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'MapCoder [14]', 87.2, 69.6]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'MGDebugger [35]', 86.8, 64.8]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'LPW [27]', 84.0, 65.2]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'EG-CFG (Ours)', 96.6, 73.0]
        return df

    elif benchmark == "humaneval":
        df = pd.DataFrame(data=[], columns=["Model", "Method", "HumanEval", "HumanEval-ET"])
        df.loc[len(df)] = ['GPT-4', 'Baseline LLM', 67.7, 50.6]
        df.loc[len(df)] = ['GPT-4', 'Self-Debugging [11]', 61.6, 45.8]
        df.loc[len(df)] = ['GPT-4', 'MetaGPT [26]', 85.9, pd.NA]
        df.loc[len(df)] = ['GPT-4', 'MapCoder [14]', 80.5, 70.1]
        df.loc[len(df)] = ['GPT-4', 'Self-Collaboration [44]', 90.7, 70.1]
        df.loc[len(df)] = ['GPT-4o', 'LPW [27]', 98.2, 84.8]
        df.loc[len(df)] = ['LLaMA 3', 'LDB [39]', 99.4, pd.NA]
        df.loc[len(df)] = ['Claude-Sonnet-3.5', 'QualityFlow [12]', 98.8, pd.NA]
        df.loc[len(df)] = ['CodeQwen1.5', 'MGDebugger [35]', 91.5, pd.NA]
        df.loc[len(df)] = ['DeepSeek-Coder-V2-Lite', 'MGDebugger [35]', 94.5, pd.NA]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'Baseline LLM', 82.92, 79.20]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'MapCoder [14]', 96.95, 81.70]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'MGDebugger [35]', 87.20, 81.09]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'LPW [27]', 95.12, 84.74]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'EG-CFG (Ours)', 99.4, 89.02]
        return df

    elif benchmark == "codecontests":
        df = pd.DataFrame(data=[], columns=["Model", "Method", "CodeContests"])
        df.loc[len(df)] = ['GPT-3.5 Turbo', 'Self-Planning [46]', 6.10]
        df.loc[len(df)] = ['GPT-3.5 Turbo', 'Analogical [43]', 7.30]
        df.loc[len(df)] = ['GPT-4', 'Self-Planning [46]', 10.90]
        df.loc[len(df)] = ['GPT-4', 'Analogical [43]', 10.90]
        df.loc[len(df)] = ['GPT-3.5 Turbo', 'MapCoder [14]', 12.70]
        df.loc[len(df)] = ['GPT-3.5 Turbo', 'CodeSim [40]', 16.40]
        df.loc[len(df)] = ['MoTCoder-15B', 'MoTCoder [45]', 26.34]
        df.loc[len(df)] = ['GPT-4', 'MapCoder [14]', 28.50]
        df.loc[len(df)] = ['GPT-4', 'CodeSim [40]', 29.10]
        df.loc[len(df)] = ['GPT-4o', 'LDB [40]', 29.30]
        df.loc[len(df)] = ['GPT-4o', 'LPW [27]', 34.70]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'Baseline LLM', 41.81]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'MapCoder [14]', 50.30]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'CodeSim [40]', 52.72]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'EG-CFG (Ours)', 60.6]
        return df

    elif benchmark == "ds1000":
        df = pd.DataFrame(data=[], columns=["Model", "Method", "DS-1000"])
        df.loc[len(df)] = ['GPT-3.5 Turbo', 'DocPrompting', 45.50]
        df.loc[len(df)] = ['GPT-3.5 Turbo', 'Self Debugging', 53.0]
        df.loc[len(df)] = ['GPT-3.5 Turbo', 'SelfEvolve', 57.1]
        df.loc[len(df)] = ['Qwen2-72B-Instruct', 'Baseline LLM', 52.8]
        df.loc[len(df)] = ['DeepSeek-Coder-V2-SFT', 'Baseline LLM', 53.2]
        df.loc[len(df)] = ['Claude 3.5 Sonnet', 'Baseline LLM', 54.3]
        df.loc[len(df)] = ['GPT-4o', 'Baseline LLM', 59.9]
        df.loc[len(df)] = ['GPT-4', 'Baseline LLM', 60.2]
        df.loc[len(df)] = ['GPT-4', 'CONLINE', 68.0]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'Baseline LLM', 38.9]
        df.loc[len(df)] = ['DeepSeek-V3-0324', 'EG-CFG (Ours)', 69.9]
        return df

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def plot_benchmark(benchmark: str):
    df = build_dataframe(benchmark)
    sns.set(style="whitegrid", context="paper", font_scale=0.85)

    df['Model'] = df['Model'].apply(lambda x: x.replace('DeepSeek-V3-0324', 'DeepSeek-V3'))
    df['Method'] = df['Method'].apply(lambda x: x.split(' [')[0].split(' (Ours)')[0])
    df['Model'] = df['Model'] + '\n' + df['Method']
    df.drop(columns=['Method'], inplace=True)

    # Configure chart options
    if benchmark == "mbpp":
        value_vars = ['MBPP', 'MBPP-ET']
        y_label = 'MBPP Score'
        ylim = (40, 110)
        colors = {'MBPP': '#94dcf2', 'MBPP-ET': '#88f2a2'}
    elif benchmark == "humaneval":
        value_vars = ['HumanEval', 'HumanEval-ET']
        y_label = 'HumanEval Score'
        ylim = (40, 110)
        colors = {'HumanEval': '#f5a158', 'HumanEval-ET': '#7069ff'}
    elif benchmark == "codecontests":
        value_vars = ['CodeContests']
        y_label = 'Accuracy (%)'
        ylim = (0, 70)
        colors = {'CodeContests': '#3F51B5'}
    elif benchmark == "ds1000":
        value_vars = ['DS-1000']
        y_label = 'DS-1000 Score'
        ylim = (30, 80)
        colors = {'DS-1000': '#8856a7'}
    else:
        return

    df_melt = df.melt(id_vars='Model', value_vars=value_vars, var_name='EvalType', value_name='Score')

    fig, ax = plt.subplots(figsize=(14, 3))
    barplot = sns.barplot(
        x="Model", y="Score", hue="EvalType", data=df_melt, width=0.7,
        palette=colors, dodge=True, gap=0.2, ax=ax, order=df['Model'].unique()
    )

    plt.xticks(rotation=55, ha='right', fontsize=13)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('center')

    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.1f', padding=4, fontsize=11)

    sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['bottom'].set_color('#2f3333')
    plt.ylim(*ylim)
    plt.xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    plt.grid(False)
    ax.legend(frameon=False, title=None, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=12)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{benchmark.upper()}_{timestamp}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved: {filename}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark performance bar plots.")
    parser.add_argument(
        "--graph",
        choices=["mbpp", "humaneval", "codecontests", "ds1000", "all"],
        required=True,
        help="Which graph to generate (or 'all' for all graphs)."
    )
    args = parser.parse_args()

    benchmarks = (
        ["mbpp", "humaneval", "codecontests", "ds1000"]
        if args.graph == "all"
        else [args.graph]
    )

    for b in benchmarks:
        plot_benchmark(b)


if __name__ == "__main__":
    main()

