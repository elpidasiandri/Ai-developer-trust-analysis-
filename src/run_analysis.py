#!/usr/bin/env python3
"""Reproducible exploratory analysis of Stack Overflow Developer Survey 2026.

Source: https://github.com/StackExchange/Survey/tree/main/packages/archive/2026
Target: AITrust (five substantive response categories; 'not sure/non-use' excluded).
Purpose: non-causal association and out-of-sample predictive assessment.

Usage: python PX27_Executed_ML_Analysis.py /path/to/survey_results_public.csv --output /path/to/results
"""
import argparse
import json
from pathlib import Path
import warnings
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

SEED=42
TRUST_LABEL={
 'I do not trust it for most work tasks':'Distrust most tasks',
 'I trust it for low-risk work tasks only':'Low-risk only',
 'I trust it when I can easily verify the output':'Trust if verifiable',
 'I trust it for many tasks, but not important work decisions':'Many, not critical',
 'I trust it for many tasks, including important work decisions':'Including critical',
}
CLASSES=list(TRUST_LABEL.values())
NUM=['WorkExp']
DEMOCAT=['Age','CodeMain','Employment','Role','ICorPM','OrgSize']
AIFREQ=['AIFreq_1','AIFreq_2','AIFreq_3','AIFreq_4']
USECAT=AIFREQ+['AIDaily']
TOOLVALUES={
 'Uses coding assistants':'AI coding assistants or coding agents',
 'Uses chat tools':'General-purpose AI chat tools',
 'Uses agents/workflows':'AI agents or automated workflows',
 'Uses internal AI':'Internal AI tools built by my company',
}
USEBOOL=list(TOOLVALUES)
ALL_FEATURES=NUM+DEMOCAT+USECAT+USEBOOL

plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans', 'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold', 'figure.dpi':150})

def prepare(path):
    cols=['ResponseId','AITrust','AISelect']+NUM+DEMOCAT+USECAT
    df=pd.read_csv(path,usecols=cols,low_memory=False)
    d=df.loc[df['AITrust'].isin(TRUST_LABEL)].copy()
    d['trust_label']=d['AITrust'].map(TRUST_LABEL)
    for feat, token in TOOLVALUES.items():
        # Blank AISelect is a skip, not necessarily confirmed non-use; missingness retained.
        a=d['AISelect'].str.contains(token,regex=False,na=False).astype(float)
        a.loc[d['AISelect'].isna()]=np.nan
        d[feat]=a
    d['WorkExp']=pd.to_numeric(d['WorkExp'],errors='coerce').clip(lower=0,upper=65)
    for c in DEMOCAT+USECAT:
        d[c]=d[c].fillna('Not reported').astype('string').str.strip().fillna('Not reported').astype(str)
    return df,d

def make_preprocessor(features):
    nums=[f for f in features if f in NUM+USEBOOL]
    cats=[f for f in features if f in DEMOCAT+USECAT]
    return ColumnTransformer([
        ('numerical',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),
                               ('scale',StandardScaler())]),nums),
        ('categorical',Pipeline([('one_hot',OneHotEncoder(handle_unknown='ignore',min_frequency=30))]),cats),
    ],remainder='drop',sparse_threshold=0.3)

def model_configs():
    demofeatures=NUM+DEMOCAT
    usagefeatures=USECAT+USEBOOL
    allfeats=demofeatures+usagefeatures
    def lr(cols):
        return Pipeline([('preprocess',make_preprocessor(cols)),
          ('model', LogisticRegression(max_iter=1500,class_weight='balanced',C=1.0,random_state=SEED))])
    return {
      'Majority baseline':(allfeats,DummyClassifier(strategy='most_frequent')),
      'Developer background (LR)':(demofeatures,lr(demofeatures)),
      'AI usage (LR)':(usagefeatures,lr(usagefeatures)),
      'Combined (LR)':(allfeats,lr(allfeats)),
      'Combined (RF)':(allfeats,Pipeline([('preprocess',make_preprocessor(allfeats)),
         ('model',RandomForestClassifier(n_estimators=160,max_depth=18,min_samples_leaf=10,
                        max_features='sqrt',class_weight='balanced_subsample',n_jobs=2,random_state=SEED))]))
    }

def bootstrap_metric_ci(y,p,n_boot=400):
    rng=np.random.default_rng(3847)
    y=np.asarray(y);p=np.asarray(p);val=[]
    for _ in range(n_boot):
        ix=rng.integers(0,len(y),len(y))
        val.append(f1_score(y[ix],p[ix],labels=CLASSES,average='macro',zero_division=0))
    return np.quantile(val,[.025,.975]).tolist()

def figure_model_comparison(metrics,out):
    metrics=metrics.iloc[::-1]
    fig,ax=plt.subplots(figsize=(8.1,3.5))
    y=np.arange(len(metrics)); bars=ax.barh(y,metrics['Holdout macro-F1'],color='#37638a',height=.63)
    for i,v in enumerate(metrics['Holdout macro-F1']):
        ax.text(v+.008,i,f'{v:.3f}',va='center',fontsize=9)
    ax.set_yticks(y,metrics['Model']);ax.set_xlim(0,max(metrics['Holdout macro-F1'])*1.25)
    ax.set_xlabel('Held-out macro-F1 (higher is better)'); ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    ax.set_title('Five-category AI trust prediction — held-out test set',loc='left',pad=13)
    fig.tight_layout();fig.savefig(out,dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig)

def figure_trust_experience(d,out):
    t=d.copy();t['Experience band']=pd.cut(t['WorkExp'],[-.01,2,5,10,20,65],labels=['0–2 years','3–5 years','6–10 years','11–20 years','21+ years'])
    t=t[t['Experience band'].notna()]
    gb=t.groupby('Experience band',observed=True)['trust_label']
    summary=[]
    for grp,a in gb:
        n=len(a); ver=(a=='Trust if verifiable').sum(); high=(a=='Including critical').sum()
        ci=proportion_confint(ver,n,method='wilson')
        summary.append({'Experience band':str(grp),'n':int(n),'Verifiable trust n':int(ver),'Verifiable trust %':100*ver/n,
                         '95% CI lower %':100*ci[0],'95% CI upper %':100*ci[1],
                         'Critical-decision trust n':int(high),'Critical-decision trust %':100*high/n})
    s=pd.DataFrame(summary)
    fig,ax=plt.subplots(figsize=(7.8,3.7))
    yy=np.arange(len(s));pct=s['Verifiable trust %'].values
    ax.errorbar(pct,yy,xerr=[pct-s['95% CI lower %'].values,s['95% CI upper %'].values-pct],
                fmt='o',markersize=7,capsize=4,color='#37638a',lw=1.5)
    ax.set_yticks(yy,[f"{lab}  (n={n:,})" for lab,n in zip(s['Experience band'],s.n)])
    ax.invert_yaxis();ax.set_xlim(39,62.5)
    ax.set_xlabel('Share reporting verification-contingent trust (%)')
    ax.set_title('Verification-contingent trust by professional experience',loc='left',pad=12)
    ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
    for i,v in enumerate(pct): ax.text(s['95% CI upper %'].iloc[i]+.75,i,f'{v:.1f}%',va='center')
    fig.tight_layout();fig.savefig(out,dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig)
    return s

def permutation_by_group(model, Xte, yte, features, out, n_repeats=5):
    # Entire original feature column permuted before pipeline, so one-hot groups stay intact.
    # Relative metric decreases are interpretable as model reliance, not causal effects.
    rng=np.random.default_rng(2026)
    base=f1_score(yte,model.predict(Xte),labels=CLASSES,average='macro',zero_division=0)
    details=[]
    for c in features:
        values=[]
        for _ in range(n_repeats):
            copy=Xte.copy();copy[c]=rng.permutation(Xte[c].to_numpy())
            p=model.predict(copy)
            values.append(base-f1_score(yte,p,labels=CLASSES,average='macro',zero_division=0))
        details.append({'Predictor':c,'Mean macro-F1 drop':float(np.mean(values)),
                        'SD macro-F1 drop':float(np.std(values,ddof=1))})
    f=pd.DataFrame(details).sort_values('Mean macro-F1 drop',ascending=False)
    ss=f.head(10).iloc[::-1].copy()
    nice={'AIDaily':'Daily AI-use duration','AIFreq_1':'Coding-assistant frequency',
          'AIFreq_2':'Chat-tool frequency','AIFreq_3':'Agent/workflow frequency',
          'AIFreq_4':'Internal-AI frequency','OrgSize':'Organization size',
          'ICorPM':'Individual contributor / manager','CodeMain':'Coding responsibility',
          'Uses chat tools':'Uses AI chat tools','Uses internal AI':'Uses internal AI tools'}
    ss['Predictor']=ss['Predictor'].map(lambda x:nice.get(x,x))
    fig,ax=plt.subplots(figsize=(7.8,4.2));colors=['#37638a' if x>0 else '#a4a4a4' for x in ss['Mean macro-F1 drop']]
    ax.barh(ss.Predictor,ss['Mean macro-F1 drop'],color=colors)
    ax.axvline(0,color='#212c38',lw=.75);ax.set_xlabel('Decrease in held-out macro-F1 after shuffling predictor')
    ax.set_title('Predictive importance of individual survey features',loc='left',pad=12)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(out,dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig)
    return f

def main():
    ap=argparse.ArgumentParser();ap.add_argument('csv',type=Path);ap.add_argument('--output',type=Path,default=Path('PX27_ML_Results'))
    ap.add_argument('--code-primary',action='store_true',help='Sensitivity analysis: only respondents who create code as a primary part of their job')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    raw,d=prepare(args.csv)
    if args.code_primary: d=d.loc[d['CodeMain']=='I create code as a primary part of my job'].copy()
    assert len(raw)>1000 and len(d)>500 and d['trust_label'].nunique()==5
    y=d.trust_label;X=d[ALL_FEATURES].copy()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    results=[]; cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    preds={}; trained={}
    for name,(features,pipe) in model_configs().items():
        print('Training:',name,flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',category=FutureWarning)
            cvs=cross_validate(pipe,Xtr[features],ytr,cv=cv,scoring={'macro_f1':'f1_macro','balanced_acc':'balanced_accuracy'},n_jobs=1)
            pipe.fit(Xtr[features],ytr)
        p=pipe.predict(Xte[features]);preds[name]=p;trained[name]=(pipe,features)
        bounds=bootstrap_metric_ci(yte,p)
        results.append({'Model':name,'Train n':len(ytr),'Holdout n':len(yte),
            'CV macro-F1 mean':cvs['test_macro_f1'].mean(),'CV macro-F1 SD':cvs['test_macro_f1'].std(ddof=1),
            'Holdout macro-F1':f1_score(yte,p,labels=CLASSES,average='macro',zero_division=0),
            'Holdout balanced accuracy':balanced_accuracy_score(yte,p),
            'Holdout accuracy':accuracy_score(yte,p),'Bootstrap CI lower':bounds[0],'Bootstrap CI upper':bounds[1]})
        pd.DataFrame(classification_report(yte,p,labels=CLASSES,output_dict=True,zero_division=0)).T.to_csv(out/(name.replace(' ','_').replace('(','').replace(')','')+'_classification.csv'))
    m=pd.DataFrame(results);m.to_csv(out/'Table_Model_Results.csv',index=False)
    figure_model_comparison(m,out/'Figure_3_ML_Comparison.png')
    cmat=pd.DataFrame(confusion_matrix(yte,preds['Combined (LR)'],labels=CLASSES),index=CLASSES,columns=CLASSES)
    cmat.to_csv(out/'Table_Confusion_Matrix_Combined_LR.csv')
    # descriptive analysis: not ML, full available sample with observed WorkExp
    e=figure_trust_experience(d,out/'Figure_2_Trust_by_Experience.png');e.to_csv(out/'Table_Trust_by_Experience.csv',index=False)
    ev=d.copy();ev['expband']=pd.cut(ev.WorkExp,[-.01,2,5,10,20,65],labels=['0–2','3–5','6–10','11–20','21+'])
    tab=pd.crosstab(ev.expband,ev.trust_label);stat,pval,ddof,_=chi2_contingency(tab)
    cramers_v=np.sqrt(stat/(tab.values.sum()*min(tab.shape[0]-1,tab.shape[1]-1)))
    # Predefined combined logistic model for descriptive permutation assessment.
    best='Combined (LR)';pipe,features=trained[best]
    importance=permutation_by_group(pipe,Xte[features],yte,features,out/'Figure_4_Predictor_Importance.png')
    importance.to_csv(out/'Table_Permutation_Importance.csv',index=False)
    counts=d['trust_label'].value_counts().reindex(CLASSES)
    counts.to_csv(out/'Table_Trust_Response_Counts.csv',header=['Responses'])
    meta={
      'input_dataset':'Stack Overflow Developer Survey 2026, public respondent-level CSV',
      'source_url':'https://github.com/StackExchange/Survey/tree/main/packages/archive/2026',
      'dataset_rows':len(raw),'trust_answers_total':int(raw.AITrust.notna().sum()),
      'subgroup_code_primary_only':bool(args.code_primary),
      'eligible_five_class_rows':len(d), 'outcome_question': 'How much do you trust the output from AI tools or AI agents as part of your workflow?',
      'outcome_classes':counts.to_dict(), 'exclusion':'Not sure / I do not use AI tools and unanswered trust question',
      'random_seed':SEED, 'train_test':'80/20 stratified respondent-level split',
      'cv':'3-fold stratified CV on training portion only','predictors':ALL_FEATURES,
      'feature_scope':'Predeclared developer background and reported AI usage, excluding post-outcome trust/sentiment/verification answers',
      'primary_metric':'macro-F1 on untouched test set','uncertainty':'400 test-set bootstrap replicates for macro-F1 intervals',
      'experience_association':{'chi2':float(stat),'df':int(ddof),'p':float(pval),'cramers_v':float(cramers_v),'n':int(tab.values.sum())},
      'permutation_model':best,
      'limits':['self-selected, cross-sectional sample','self-reported trust, not observed correctness or programming skill',
       'unadjusted experience group comparisons may be confounded','model selection and importance are exploratory',
       'survey skip-logic and missingness may influence prediction','prediction does not establish causality']}
    (out/'Run_Metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n========== RESULTS ==========\n',m.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nTrust n:',len(d),'\n',counts.to_string())
    print('\nExperience\n',e.to_string(index=False,float_format=lambda x:f'{x:.2f}'))
    print('\nExperience chi-square',stat,pval,'Cramers V',cramers_v)
    print('\nPermutation importance\n',importance.head(10).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nFiles saved to',out)

if __name__=='__main__':main()
