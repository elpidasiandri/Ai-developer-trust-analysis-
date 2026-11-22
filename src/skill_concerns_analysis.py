#!/usr/bin/env python3
"""Reproduce skill-loss-or-replacement concern by experience from Stack Overflow Survey 2026.
Usage: python PX27_skill_concerns_analysis.py data/survey_results_public.csv --output results/Table_Skill_Concerns_By_Experience.csv

The AIAvoid item combines two motivations and must not be labelled as measured skill loss.
"""
import argparse
from pathlib import Path
import pandas as pd

TARGET = 'Concern over losing job skills or training an AI to replace you'
CODE_PRIMARY = 'I create code as a primary part of my job'
GROUPS = ['0–2 years','3–5 years','6–10 years','11–20 years','21+ years']

def run(source, output):
    data = pd.read_csv(source, usecols=['CodeMain','WorkExp','AIAvoid'], low_memory=False)
    data = data.loc[data['CodeMain'].eq(CODE_PRIMARY) & data['AIAvoid'].notna()].copy()
    data['WorkExp'] = pd.to_numeric(data['WorkExp'], errors='coerce')
    data = data.dropna(subset=['WorkExp'])
    data['Experience group'] = pd.cut(data['WorkExp'], bins=[-0.01, 2, 5, 10, 20, float('inf')], labels=GROUPS)
    data['Selected skill/replacement concern'] = data['AIAvoid'].eq(TARGET)
    summary = data.groupby('Experience group', observed=True)['Selected skill/replacement concern'].agg(respondents='count', selected='sum').reset_index()
    summary['percentage'] = 100 * summary['selected'] / summary['respondents']
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print('Eligible code-primary respondents:', len(data))
    print(summary.to_string(index=False, formatters={'percentage': lambda v: f'{v:.1f}%'}))
    return summary

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('source', help='Official survey_results_public.csv')
    ap.add_argument('--output', default='results/Table_Skill_Concerns_By_Experience.csv')
    args = ap.parse_args()
    run(args.source, args.output)
