#!/usr/bin/env python3
"""Reproduce the PX/27 matched programming-task satisfaction comparison.

Source: Stack Overflow Developer Survey 2026, AISat_2/3/4 and CodeMain.
Run: python src/programming_task_satisfaction.py data/survey_results_public.csv

This is a secondary analysis of reported satisfaction, NOT experimental
measurement of productivity, comprehension, or skill change.
"""

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

TASKS = {
    "Debugging / refactoring": "AISat_4",
    "Implementation": "AISat_2",
    "Code review": "AISat_3",
}
POSITIVE = {"Somewhat improved", "Greatly improved"}
CODE_PRIMARY = "I create code as a primary part of my job"


def prepare(source):
    return pd.read_csv(
        source, usecols=["CodeMain", "AISat_2", "AISat_3", "AISat_4"],
        low_memory=False,
    )


def summarize_group(data, group):
    complete = data.dropna(subset=list(TASKS.values()))
    counts = []
    for task, col in TASKS.items():
        positives = complete[col].isin(POSITIVE)
        counts.append({
            "group": group,
            "task": task,
            "respondents": int(len(complete)),
            "improved_count": int(positives.sum()),
            "improved_pct": float(100 * positives.mean()),
        })

    baseline = complete[TASKS["Debugging / refactoring"]].isin(POSITIVE).astype(int).to_numpy()
    contrasts = []
    for task in ["Implementation", "Code review"]:
        other = complete[TASKS[task]].isin(POSITIVE).astype(int).to_numpy()
        paired_delta = baseline - other
        n = len(paired_delta)
        gap = 100 * paired_delta.mean()
        se = 100 * paired_delta.std(ddof=1) / np.sqrt(n)
        discordant = int((paired_delta != 0).sum())
        debugging_only = int((paired_delta > 0).sum())
        contrasts.append({
            "group": group,
            "comparison": f"Debugging / refactoring vs {task}",
            "respondents": n,
            "difference_pp": float(gap),
            "ci95_lower_pp": float(gap - 1.96 * se),
            "ci95_upper_pp": float(gap + 1.96 * se),
            "discordant_pairs": discordant,
            "debugging_only_improved": debugging_only,
            "exact_binomial_p": float(binomtest(debugging_only, n=discordant, p=0.5).pvalue),
        })
    return counts, contrasts


def render_plot(summary, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    chosen = summary[summary['group'].eq('code_primary')].set_index('task')
    tasks = list(TASKS)
    values = [chosen.loc[k, 'improved_pct'] for k in tasks]
    n = int(chosen.iloc[0]['respondents'])
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    colors = ['#2c5267', '#8399a2', '#abb4ba']
    bars = ax.barh(tasks[::-1], values[::-1], color=colors[::-1], height=0.52,
                   edgecolor='#34434a', linewidth=.55)
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', ha='left', fontsize=11, weight='bold')
    ax.set_xlim(0,82)
    ax.set_xlabel('Respondents reporting improved task satisfaction (%)')
    ax.set_title(f'Same {n:,} respondents for all three activities', fontsize=12, pad=18)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.15)
    ax.tick_params(axis='y', length=0)
    for spine in ['top','left','right']: ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main(source, output):
    output=Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data=prepare(source)
    code_primary=data.loc[data.CodeMain.eq(CODE_PRIMARY)]
    rows_a, rows_b = [], []
    for group, selected in [('code_primary', code_primary), ('all_respondents', data)]:
        a,b=summarize_group(selected, group)
        rows_a.extend(a);rows_b.extend(b)
    table=pd.DataFrame(rows_a)
    contrasts=pd.DataFrame(rows_b)
    table.to_csv(output/'Table_Programming_Task_Satisfaction.csv',index=False)
    contrasts.to_csv(output/'Table_Paired_Programming_Task_Contrasts.csv',index=False)
    render_plot(table, output/'Figure_5_Programming_Tasks_Paired.png')
    print(table.to_string(index=False))
    print(contrasts.to_string(index=False))
    return table,contrasts

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('survey_csv', help='Official 2026 Stack Overflow survey_results_public.csv')
    parser.add_argument('--output', default='outputs/programming_tasks')
    args = parser.parse_args()
    main(args.survey_csv, args.output)
