#!/usr/bin/env python3
"""Train an exploratory AmoPaw BioAge model on a synthetic dataset.
This is a pipeline demonstration, not a validated biological-age model.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

CAT = ["物种", "性别", "绝育状态", "品种"]
NUM = ["体重_kg", "呼吸率中位数", "呼吸率中位数_波动", "呼吸率中位数_有效率",
       "心率中位数", "心率中位数_波动", "心率中位数_有效率",
       "HRV_RMSSD中位数", "HRV_RMSSD中位数_波动", "HRV_RMSSD中位数_有效率",
       "高信号质量占比", "有效测量占比", "有效记录数"]
FEATURES = CAT + NUM

def aggregate(raw):
    rows=[]
    for pid,g in raw.groupby("pet_id", sort=False):
        r={"pet_id":pid}
        for c in ["物种","性别","绝育状态","品种","年龄_岁","建议分组_按宠物ID"]:
            s=g[c].dropna(); r[c]=s.iloc[0] if len(s) else np.nan
        r["体重_kg"]=pd.to_numeric(g["体重_kg"],errors="coerce").median()
        for c,n in [("呼吸率_次每分钟","呼吸率中位数"),("心率_bpm","心率中位数"),("HRV_RMSSD_ms","HRV_RMSSD中位数")]:
            v=pd.to_numeric(g[c],errors="coerce")
            r[n]=v.median(); r[n+"_波动"]=v.std(ddof=0) if v.notna().sum()>1 else np.nan
            r[n+"_有效率"]=v.notna().mean()
        r["高信号质量占比"]=(g["信号质量"].astype(str)=="高").mean()
        r["有效测量占比"]=(g["测量状态"].astype(str).str.contains("安静|睡眠",regex=True)).mean()
        r["有效记录数"]=int(g[["呼吸率_次每分钟","心率_bpm","HRV_RMSSD_ms"]].notna().any(axis=1).sum())
        rows.append(r)
    return pd.DataFrame(rows)

def catprep(frame, medians):
    x=frame[FEATURES].copy()
    for c in CAT: x[c]=x[c].fillna("未知").astype(str)
    for c in NUM:
        x[c]=pd.to_numeric(x[c],errors="coerce").fillna(medians.get(c,0.0))
    return x

def score(y,p):
    return {"n_pets":int(len(y)),"MAE_years":round(float(mean_absolute_error(y,p)),4),
            "RMSE_years":round(float(np.sqrt(mean_squared_error(y,p))),4),
            "R2":round(float(r2_score(y,p)),4) if len(y)>1 else None}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="AmoPaw_BioAge_synthetic_dataset.csv")
    ap.add_argument("--out",default="artifacts")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.data); pets=aggregate(raw)
    pets["年龄_岁"]=pd.to_numeric(pets["年龄_岁"],errors="coerce")
    labeled=pets[pets["年龄_岁"].notna()].copy()
    tr=labeled[labeled["建议分组_按宠物ID"]=="训练集"]
    va=labeled[labeled["建议分组_按宠物ID"]=="验证集"]
    te=labeled[labeled["建议分组_按宠物ID"]=="测试集"]
    if min(len(tr),len(va),len(te))==0: raise ValueError("训练/验证/测试至少一个集合为空")
    medians={c:float(tr[c].median()) if tr[c].notna().any() else 0.0 for c in NUM}
    # Ridge baseline
    pre=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),NUM),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),CAT)])
    ridge=Pipeline([("preprocess",pre),("model",Ridge(alpha=10.0))]).fit(tr[FEATURES],tr["年龄_岁"])
    ridge_val=ridge.predict(va[FEATURES])
    # CatBoost candidate
    cb=CatBoostRegressor(iterations=500,depth=5,learning_rate=.03,loss_function="MAE",
                         random_seed=42,verbose=False,l2_leaf_reg=5.0)
    cat_idx=[FEATURES.index(c) for c in CAT]
    cb.fit(catprep(tr,medians),tr["年龄_岁"],cat_features=cat_idx,
           eval_set=(catprep(va,medians),va["年龄_岁"]),early_stopping_rounds=60,verbose=False)
    cb_val=cb.predict(catprep(va,medians))
    selected="CatBoost" if mean_absolute_error(va["年龄_岁"],cb_val)<mean_absolute_error(va["年龄_岁"],ridge_val) else "Ridge"
    # Evaluate untouched test
    test_pred=cb.predict(catprep(te,medians)) if selected=="CatBoost" else ridge.predict(te[FEATURES])
    report={"warning":"Synthetic dataset only; metrics are not real-world validation.",
            "counts":{"pets_total":int(pets.pet_id.nunique()),"labeled":int(len(labeled)),
                      "train":int(len(tr)),"validation":int(len(va)),"test":int(len(te))},
            "validation":{"Ridge":score(va["年龄_岁"],ridge_val),"CatBoost":score(va["年龄_岁"],cb_val)},
            "selected_by_validation_MAE":selected,"held_out_test":score(te["年龄_岁"],test_pred)}
    (out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    tv=pd.concat([tr,va],ignore_index=True)
    if selected=="CatBoost":
        final=CatBoostRegressor(iterations=max(100,int(cb.best_iteration_+1) if cb.best_iteration_ is not None and cb.best_iteration_>=0 else 300),
            depth=5,learning_rate=.03,loss_function="MAE",random_seed=42,verbose=False,l2_leaf_reg=5.0)
        final.fit(catprep(tv,medians),tv["年龄_岁"],cat_features=cat_idx,verbose=False)
        final.save_model(str(out/"bioage_catboost.cbm"))
        import joblib
        joblib.dump({"kind":"CatBoostRegressor","feature_cols":FEATURES,"cat_features":CAT,
                     "num_features":NUM,"imputation_medians":medians,"model_file":"bioage_catboost.cbm"},
                    out/"model_metadata.joblib")
    else:
        final=Pipeline([("preprocess",pre),("model",Ridge(alpha=10.0))]).fit(tv[FEATURES],tv["年龄_岁"])
        import joblib
        joblib.dump(final,out/"bioage_model.joblib")
        joblib.dump({"kind":"sklearn_pipeline","model":final,"feature_cols":FEATURES,
                     "cat_features":CAT,"num_features":NUM},out/"model_metadata.joblib")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
