# -*- coding: utf-8 -*-
"""
聚焦生活满意度的完整分析
基于CLASS 2020 + 第九章分布检验 + 第四章Logistic回归
"""
import pandas as pd; import numpy as np
import json, os, warnings; warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm, chi2, kstest, shapiro, ks_2samp, mannwhitneyu, kruskal, spearmanr
import statsmodels.api as sm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei']; plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 数据加载
# ============================================================
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_BASE, 'data')
_FIG = os.path.join(_BASE, 'figures')
_RES = os.path.join(_BASE, 'results')
for _d in (_DATA, _FIG, _RES):
    os.makedirs(_d, exist_ok=True)

df = pd.read_excel(os.path.join(_DATA, 'CLASS2020_清洗后问卷数据.xlsx'))
N = len(df)
df['BMI'] = df['G1_2：体重'] / ((df['G1_1：身高'] / 100) ** 2)
df['health_poor'] = (df['B1：自评健康'] >= 3).astype(int)
df['life_dissat'] = (df['B17：生活满意度'] >= 3).astype(int)
df['female'] = (df['A1：性别'] == 2).astype(int)
df['childhood_hunger'] = (df['B14：童年是否经常饿着睡觉'] == 2).astype(int)
df['internet'] = df['D18：是否上网']
df['has_spouse'] = df['是否有配偶']

fig_dir = _FIG
os.makedirs(fig_dir, exist_ok=True)
R = {}

def fmt_p(p_val):
    return '<0.001' if p_val < 0.001 else '%.4f' % p_val

# ============================================================
# 1. 生活满意度基本分布特征
# ============================================================
print('=== 1. 生活满意度基本分布 ===')
ls = df['B17：生活满意度']
ls_labels = ['很满意','比较满意','一般','比较不满意','很不满意']
ls_freq = ls.value_counts().sort_index()
ls_pct = (ls_freq / N * 100).round(1)

print(f'生活满意度均值={ls.mean():.4f}, 标准差={ls.std():.4f}')
print(f'偏度={ls.skew():.4f}, 峰度={ls.kurtosis():.4f}')
print(f'满意比例(1-2): {(ls<=2).mean()*100:.1f}%')
print(f'不满意比例(3-5): {(ls>=3).mean()*100:.1f}%')
R['ls_desc'] = {
    'mean': float(ls.mean()), 'std': float(ls.std()),
    'skewness': float(ls.skew()), 'kurtosis': float(ls.kurtosis()),
    'N': N,
    'freq': {str(k): int(v) for k,v in ls_freq.items()},
    'pct': {str(k): float(v) for k,v in ls_pct.items()},
    'pct_satisfied': float((ls<=2).mean()*100),
    'pct_dissatisfied': float((ls>=3).mean()*100),
}

# ============================================================
# 2. 第九章方法: 分布检验（聚焦生活满意度）
# ============================================================
print('\n=== 2. 分布检验 (第九章) ===')

# 2.1 K-S检验: 生活满意度总体分布 vs 理论分布
print('--- 2.1 K-S检验 ---')
# Test vs uniform
unique_vals = sorted(ls.unique())
n_cat = len(unique_vals)
# K-S for discrete: test vs uniform(1,5)
d_ks, p_ks = kstest(ls, 'uniform', args=(1, 4))  # uniform on [1,5]
print(f'K-S(生活满意度~均匀[1,5]): D={d_ks:.4f}, p={p_ks:.6f}')

# Also test specific distributions
# vs normal with sample parameters
d_norm, p_norm = kstest((ls - ls.mean())/ls.std(), 'norm')
print(f'K-S(生活满意度~正态): D={d_norm:.4f}, p={p_norm:.6f}')

R['ks_ls'] = {
    'D_uniform': float(d_ks), 'p_uniform': float(p_ks),
    'D_normal': float(d_norm), 'p_normal': float(p_norm),
}

# 2.2 Pearson χ² GOF: 生活满意度均匀分布检验
print('--- 2.2 Pearson χ² GOF ---')
obs = ls_freq.values.astype(float)
exp_uniform = np.ones(n_cat) * N / n_cat
chi2_ls = np.sum((obs - exp_uniform)**2 / exp_uniform)
p_chi2_ls = 1 - chi2.cdf(chi2_ls, n_cat - 1)
print(f'χ²={chi2_ls:.1f}, df={n_cat-1}, p={p_chi2_ls:.6f}')

# Test specific expected distribution (observed proportions as expected)
# H0: p1=17.5%, p2=47.9%, p3=25.8%, p4=7.1%, p5=1.7% (from sample)
R['pearson_ls'] = {
    'chi2': float(chi2_ls), 'df': n_cat-1, 'p_value': float(p_chi2_ls),
    'observed': [int(x) for x in obs],
    'expected_uniform': [float(x) for x in exp_uniform],
}

# 2.3 Shapiro-Wilk: 生活满意度正态性
print('--- 2.3 Shapiro-Wilk ---')
w_ls, p_w_ls = shapiro(ls.sample(min(5000, N), random_state=42))
print(f'Shapiro-Wilk(生活满意度): W={w_ls:.4f}, p={p_w_ls:.6f}')
R['sw_ls'] = {'W': float(w_ls), 'p_value': float(p_w_ls)}

# 2.4 双样本K-S检验: 生活满意度分组比较
print('--- 2.4 双样本K-S检验 ---')
ks2_results = {}
group_comparisons = [
    ('internet', '上网', {0:'不上网', 1:'上网'}),
    ('female', '性别', {0:'男', 1:'女'}),
    ('health_poor', '自评健康', {0:'健康好', 1:'健康差'}),
    ('has_spouse', '配偶', {0:'无配偶', 1:'有配偶'}),
    ('childhood_hunger', '童年饥饿', {0:'否', 1:'是'}),
]
for gvar, gname, labels in group_comparisons:
    g0 = df[df[gvar]==0]['B17：生活满意度'].dropna()
    g1 = df[df[gvar]==1]['B17：生活满意度'].dropna()
    d, p = ks_2samp(g0, g1)
    ks2_results[f'{gname}({labels[0]}vs{labels[1]})'] = {
        'D': float(d), 'p_value': float(p),
        'g0_mean': float(g0.mean()), 'g1_mean': float(g1.mean()),
        'g0_n': len(g0), 'g1_n': len(g1),
    }
    print(f'{gname}: D={d:.4f}, p={p:.6f}, mean0={g0.mean():.3f}, mean1={g1.mean():.3f}')
R['ks2_ls'] = ks2_results

# ============================================================
# 3. Wilcoxon秩和检验 + Kruskal-Wallis H检验
# ============================================================
print('\n=== 3. 秩检验 ===')

# 3.1 Wilcoxon (两组)
mw_results = {}
for gvar, gname, labels in group_comparisons:
    g0 = df[df[gvar]==0]['B17：生活满意度']
    g1 = df[df[gvar]==1]['B17：生活满意度']
    u, p = mannwhitneyu(g1, g0, alternative='two-sided')
    n0, n1 = len(g0), len(g1)
    z = (u - n0*n1/2) / np.sqrt(n0*n1*(n0+n1+1)/12)
    r = abs(z) / np.sqrt(n0+n1)
    mw_results[f'{gname}({labels[0]}vs{labels[1]})'] = {
        'U': float(u), 'Z': float(z), 'p_value': float(p),
        'effect_size_r': float(r),
        'g0_mean': float(g0.mean()), 'g1_mean': float(g1.mean()),
        'g0_n': n0, 'g1_n': n1,
    }
    print(f'{gname}: U={u:.0f}, Z={z:.3f}, p={p:.6f}, r={r:.4f}')
R['wilcoxon_ls'] = mw_results

# 3.2 Kruskal-Wallis (多组)
kw_results = {}
for gvar, gname in [('A3：文化程度','文化程度'), ('G4：锻炼强度','锻炼强度'),
                     ('B1：自评健康','自评健康'), ('D12_1：社交网络（家人/朋友）','社交网络')]:
    groups = [grp['B17：生活满意度'].dropna().values for _, grp in df.groupby(gvar)]
    h, p = kruskal(*groups)
    kw_results[gname] = {'H': float(h), 'p_value': float(p), 'n_groups': len(groups)}
    # mean rank per group
    all_vals = np.concatenate(groups)
    all_ranks = stats.rankdata(all_vals)
    offsets = np.cumsum([0] + [len(g) for g in groups[:-1]])
    mean_ranks = {i: float(all_ranks[offsets[i]:offsets[i]+len(g)].mean()) for i, g in enumerate(groups)}
    kw_results[gname]['mean_ranks'] = mean_ranks
    kw_results[gname]['group_means'] = {i: float(g.mean()) for i, g in enumerate(groups)}
    kw_results[gname]['group_sizes'] = {i: len(g) for i, g in enumerate(groups)}
    print(f'{gname}: H={h:.3f}, df={len(groups)-1}, p={p:.6f}')
R['kruskal_wallis'] = kw_results

# 3.3 Spearman秩相关
print('--- Spearman相关 ---')
spearman_vars = {
    '年龄':'年龄', 'BMI':'BMI', 'A3：文化程度':'文化程度',
    'B1：自评健康':'自评健康', 'D12_1：社交网络（家人/朋友）':'社交网络',
    'G4：锻炼强度':'锻炼强度', 'internet':'上网', 'female':'女性',
    'has_spouse':'有配偶', 'childhood_hunger':'童年饥饿'
}
spearman_results = {}
for var, name in spearman_vars.items():
    rs, p = spearmanr(df[var], ls)
    spearman_results[name] = {'rs': float(rs), 'p_value': float(p)}
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    print(f'{name}: rs={rs:.4f}, p={p:.6f} {sig}')
R['spearman'] = spearman_results

# ============================================================
# 4. Logistic回归（第四章）
# ============================================================
print('\n=== 4. Logistic回归 ===')

# 模型: 生活不满意(life_dissat)的决定因素
X_vars = ['female','年龄','BMI','A3：文化程度','health_poor','childhood_hunger',
          'D12_1：社交网络（家人/朋友）','has_spouse','internet','G4：锻炼强度']
X_raw = df[X_vars].copy()
X = sm.add_constant(X_raw)
y = df['life_dissat']
Xs = X.copy()
for v in ['年龄','BMI']:
    Xs[v] = (Xs[v] - Xs[v].mean()) / Xs[v].std()

model = sm.Logit(y, Xs.astype(float)).fit(disp=0)
print(model.summary().tables[1])
print(f'LogLik={model.llf:.2f}, PseudoR2={model.prsquared:.4f}, AIC={model.aic:.2f}')

var_labels = {
    'const':'截距项','female':'女性(vs男性)','年龄':'年龄(标准化)','BMI':'BMI(标准化)',
    'A3：文化程度':'文化程度','health_poor':'自评健康差(vs好)',
    'childhood_hunger':'童年饥饿(是vs否)',
    'D12_1：社交网络（家人/朋友）':'社交网络','has_spouse':'有配偶(vs无)',
    'internet':'上网(vs不上网)','G4：锻炼强度':'锻炼强度',
}
coefs = []
params = model.params; se = model.bse; zvals = model.tvalues; pvals = model.pvalues
ci = model.conf_int()
for v in params.index:
    coefs.append({
        'variable': v, 'label': var_labels.get(v, v),
        'beta': float(params[v]), 'se': float(se[v]),
        'z': float(zvals[v]), 'p': float(pvals[v]),
        'OR': float(np.exp(params[v])),
        'CI_low': float(np.exp(ci.iloc[:,0][v])),
        'CI_high': float(np.exp(ci.iloc[:,1][v])),
        'significant': pvals[v] < 0.05,
    })

R['logit'] = {
    'coefficients': coefs,
    'loglik': float(model.llf), 'pseudo_r2': float(model.prsquared),
    'aic': float(model.aic), 'bic': float(model.bic),
    'lr_chi2': float(model.llr), 'lr_df': int(model.df_model), 'lr_p': float(model.llr_pvalue),
}

# Hosmer-Lemeshow
y_pred = model.predict(Xs.astype(float))
df_hl = pd.DataFrame({'y':y, 'p':y_pred})
df_hl['g'] = pd.qcut(df_hl['p'], q=10, duplicates='drop')
hl_stat = 0
for gname, grp in df_hl.groupby('g'):
    o1 = grp['y'].sum(); o0 = len(grp) - o1
    e1 = grp['p'].sum(); e0 = len(grp) - e1
    if e1>0 and e0>0: hl_stat += (o1-e1)**2/e1 + (o0-e0)**2/e0
df_g = len(df_hl['g'].unique()); hl_p = 1 - chi2.cdf(hl_stat, max(df_g-2,1))
R['hl'] = {'chi2': float(hl_stat), 'df': max(df_g-2,1), 'p': float(hl_p)}
print(f'H-L: chi2={hl_stat:.2f}, df={max(df_g-2,1)}, p={hl_p:.4f}')

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
Xv = df[X_vars].astype(float).dropna()
vif = {}
for i, v in enumerate(X_vars):
    vif[v] = float(variance_inflation_factor(Xv.values, i))
R['vif'] = vif

# Accuracy
pred_c = (y_pred >= 0.5).astype(int)
R['accuracy'] = {
    'accuracy': float((pred_c==y).mean()),
    'sensitivity': float(((pred_c==1)&(y==1)).sum()/y.sum()),
    'specificity': float(((pred_c==0)&(y==0)).sum()/(1-y).sum()),
}

# ============================================================
# 5. 生成图表（聚焦生活满意度）
# ============================================================
print('\n=== 5. 生成图表 ===')

# 图1: 生活满意度总体分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#2ecc71','#3498db','#f39c12','#e67e22','#e74c3c']
# Pie chart
ax = axes[0]
wedges, texts, autotexts = ax.pie(
    [ls_freq[i] for i in range(1,6)],
    labels=ls_labels,
    autopct='%1.1f%%',
    colors=colors,
    explode=(0.02,0.02,0.02,0.02,0.02),
    startangle=90)
for t in autotexts: t.set_fontsize(9)
ax.set_title('生活满意度等级分布\n(饼图)', fontsize=13, fontweight='bold')

# Bar chart
ax = axes[1]
bars = ax.bar(range(1,6), [ls_freq[i] for i in range(1,6)], color=colors, edgecolor='white')
for bar, v in zip(bars, [ls_freq[i] for i in range(1,6)]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100, str(v), ha='center', fontsize=10)
ax.set_xticks(range(1,6))
ax.set_xticklabels(ls_labels, fontsize=10)
ax.set_ylabel('频数', fontsize=11)
ax.set_title('生活满意度等级分布\n(柱状图)', fontsize=13, fontweight='bold')
ax.axhline(y=N/5, color='gray', linestyle='--', alpha=0.7, label=f'均匀分布期望({N/5:.0f})')
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir,'fig1_ls_dist.png'),dpi=150,bbox_inches='tight'); plt.close()
print('Fig1 done')

# 图2: 各分组的生活满意度分布对比（K-S检验结果）
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
group_plots = [
    (0,0,'internet','是否上网',{0:'不上网',1:'上网'}),
    (0,1,'female','性别',{0:'男',1:'女'}),
    (0,2,'health_poor','自评健康',{0:'健康好',1:'健康差'}),
    (1,0,'has_spouse','是否有配偶',{0:'无配偶',1:'有配偶'}),
    (1,1,'childhood_hunger','童年饥饿',{0:'否',1:'是'}),
    (1,2,'G4：锻炼强度','锻炼强度',{1:'低',2:'中',3:'高'}),
]
for row, col, gvar, gtitle, glabels in group_plots:
    ax = axes[row, col]
    data_list = [df[df[gvar]==k]['B17：生活满意度'].dropna().values for k in sorted(glabels.keys())]
    labels_list = [glabels[k] for k in sorted(glabels.keys())]
    bp = ax.boxplot(data_list, labels=labels_list, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], ['#3498db','#e74c3c','#2ecc71'][:len(labels_list)]):
        patch.set_facecolor(color); patch.set_alpha(0.6)

    # K-S test annotation
    if len(data_list) == 2:
        d,p = ks_2samp(data_list[0], data_list[1])
        ax.set_title(f'{gtitle}\n(K-S: D={d:.3f}, p={fmt_p(p)})', fontsize=11, fontweight='bold')
    else:
        h,p = kruskal(*data_list)
        ax.set_title(f'{gtitle}\n(KW-H={h:.2f}, p={fmt_p(p)})', fontsize=11, fontweight='bold')
    ax.set_ylabel('生活满意度(1-5)', fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir,'fig2_group_boxplot.png'),dpi=150,bbox_inches='tight'); plt.close()
print('Fig2 done')

# 图3: Kruskal-Wallis各组平均秩比较
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Education
edu_means = df.groupby('A3：文化程度')['B17：生活满意度'].mean()
edu_labels = ['不识字','私塾/扫盲','小学','初中','高中/中专','大专','本科及以上']
axes[0].bar(range(len(edu_means)), edu_means.values, color=['#e74c3c']*2+['#f39c12']*3+['#3498db']*2, edgecolor='white')
for i, v in enumerate(edu_means.values):
    axes[0].text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)
axes[0].set_xticks(range(len(edu_means)))
axes[0].set_xticklabels(edu_labels, rotation=30, fontsize=9)
axes[0].set_ylabel('生活满意度均值', fontsize=11)
axes[0].set_title(f'文化程度 vs 生活满意度均值\n(Kruskal-Wallis H={kw_results["文化程度"]["H"]:.1f}, p<0.001)', fontsize=11, fontweight='bold')
axes[0].axhline(y=ls.mean(), color='gray', linestyle='--', alpha=0.5, label=f'总均值={ls.mean():.3f}')
axes[0].legend(fontsize=9)

# Exercise
ex_means = df.groupby('G4：锻炼强度')['B17：生活满意度'].mean()
ex_labels = ['低','中','高']
axes[1].bar(range(len(ex_means)), ex_means.values, color=['#e74c3c','#f39c12','#2ecc71'], edgecolor='white')
for i, v in enumerate(ex_means.values):
    axes[1].text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)
axes[1].set_xticks(range(len(ex_means)))
axes[1].set_xticklabels(ex_labels, fontsize=10)
axes[1].set_ylabel('生活满意度均值', fontsize=11)
axes[1].set_title(f'锻炼强度 vs 生活满意度均值\n(Kruskal-Wallis H={kw_results["锻炼强度"]["H"]:.1f}, p={fmt_p(kw_results["锻炼强度"]["p_value"])})', fontsize=11, fontweight='bold')
axes[1].axhline(y=ls.mean(), color='gray', linestyle='--', alpha=0.5, label=f'总均值={ls.mean():.3f}')
axes[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir,'fig3_mean_rank.png'),dpi=150,bbox_inches='tight'); plt.close()
print('Fig3 done')

# 图4: Spearman相关热力图
fig, ax = plt.subplots(figsize=(10, 8))
corr_vars_map = {
    '年龄':'年龄','BMI':'BMI','A3：文化程度':'文化程度',
    'B1：自评健康':'自评健康','D12_1：社交网络（家人/朋友）':'社交网络',
    'G4：锻炼强度':'锻炼强度','internet':'上网','female':'女性',
    'has_spouse':'有配偶','childhood_hunger':'童年饥饿','B17：生活满意度':'生活满意度'
}
corr_labels = list(corr_vars_map.values())
corr_vars_list = list(corr_vars_map.keys())
corr_mat = df[corr_vars_list].corr(method='spearman')
sns.heatmap(corr_mat, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            xticklabels=corr_labels, yticklabels=corr_labels,
            vmin=-0.5, vmax=0.5, ax=ax, linewidths=0.5,
            annot_kws={'fontsize':8})
ax.set_title('Spearman秩相关系数矩阵\n(聚焦生活满意度)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(fig_dir,'fig4_spearman_heatmap.png'),dpi=150,bbox_inches='tight'); plt.close()
print('Fig4 done')

# 图5: Logistic回归OR森林图
fig, ax = plt.subplots(figsize=(10, 8))
coefs_sig = [c for c in coefs if c['variable'] != 'const']
var_names = [c['label'] for c in coefs_sig]
ors = [c['OR'] for c in coefs_sig]
ci_l = [c['CI_low'] for c in coefs_sig]; ci_u = [c['CI_high'] for c in coefs_sig]
pvals_c = [c['p'] for c in coefs_sig]
colors_c = ['#e74c3c' if (o>1 and p<0.05) else '#3498db' if (o<1 and p<0.05) else '#95a5a6' for o,p in zip(ors,pvals_c)]
y_pos = range(len(var_names))
ax.barh(y_pos, [o-1 for o in ors], left=1, height=0.5, color=colors_c, alpha=0.8)
ax.errorbar(ors, y_pos, xerr=[[o-l for o,l in zip(ors,ci_l)],[u-o for o,u in zip(ors,ci_u)]],
            fmt='none', ecolor='black', capsize=3)
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
ax.set_yticks(y_pos); ax.set_yticklabels(var_names, fontsize=10)
ax.set_xlabel('优势比(OR)', fontsize=12)
ax.set_title('Logistic回归: 生活不满意的影响因素\n(OR及95%CI)', fontsize=13, fontweight='bold')
ax.set_xlim(0.4, 6.2)
for i,(o,l,u,p) in enumerate(zip(ors,ci_l,ci_u,pvals_c)):
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '(ns)'
    ax.text(o+0.05, i, f'{o:.3f}{sig}', va='center', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir,'fig5_or_forest.png'),dpi=150,bbox_inches='tight'); plt.close()
print('Fig5 done')

# 图6: 上网×生活满意度交叉分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Stacked bar
ct = pd.crosstab(df['internet'], df['B17：生活满意度'])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.columns = ls_labels
ct_pct.index = ['不上网', '上网']
ct_pct.plot(kind='bar', stacked=True, ax=axes[0], color=colors, edgecolor='white')
axes[0].set_xlabel('是否上网', fontsize=11); axes[0].set_ylabel('百分比(%)', fontsize=11)
axes[0].set_title('上网状态 vs 生活满意度\n堆积百分比图', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(['不上网','上网'], rotation=0)
axes[0].legend(title='生活满意度', fontsize=8, title_fontsize=9, loc='upper right')

# KDE of life satisfaction by internet
for label, grp, color in [('不上网(N=%d)'%sum(df['internet']==0), df[df['internet']==0], '#e74c3c'),
                            ('上网(N=%d)'%sum(df['internet']==1), df[df['internet']==1], '#3498db')]:
    grp['B17：生活满意度'].plot.kde(ax=axes[1], color=color, lw=2.5, label=label)
axes[1].set_xlabel('生活满意度', fontsize=11); axes[1].set_ylabel('密度', fontsize=11)
axes[1].set_title('上网 vs 不上网老年人生活满意度核密度', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
d,p = ks_2samp(df[df['internet']==0]['B17：生活满意度'], df[df['internet']==1]['B17：生活满意度'])
axes[1].text(0.98, 0.95, f'K-S: D={d:.3f}, p<0.001\nWilcoxon: Z=-11.461', transform=axes[1].transAxes,
            ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout(); plt.savefig(os.path.join(fig_dir,'fig6_internet_ls.png'),dpi=150,bbox_inches='tight'); plt.close()
print('Fig6 done')

# 图7: 生活满意度的多维分布全景
fig = plt.figure(figsize=(16, 10))
# Heatmap: mean life satisfaction by education and health
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
# (a) Education x Health heatmap
ax1 = fig.add_subplot(gs[0, :2])
pivot = df.pivot_table(values='B17：生活满意度', index='A3：文化程度', columns='B1：自评健康', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax1,
            xticklabels=ls_labels, yticklabels=edu_labels,
            vmin=1.5, vmax=3.5, linewidths=0.5, annot_kws={'fontsize':9})
ax1.set_xlabel('自评健康', fontsize=11); ax1.set_ylabel('文化程度', fontsize=11)
ax1.set_title('生活满意度均值热力图: 文化程度 × 自评健康', fontsize=12, fontweight='bold')

# (b) Internet x Health
ax2 = fig.add_subplot(gs[0, 2])
pivot2 = df.pivot_table(values='B17：生活满意度', index='internet', columns='health_poor', aggfunc='mean')
pivot2.index = ['不上网','上网']; pivot2.columns = ['健康好','健康差']
sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax2,
            vmin=1.5, vmax=3.5, linewidths=0.5, annot_kws={'fontsize':10})
ax2.set_xlabel('自评健康', fontsize=11); ax2.set_ylabel('上网状态', fontsize=11)
ax2.set_title('上网×健康交互\n生活满意度均值', fontsize=11, fontweight='bold')

# (c) Spouse x Health
ax3 = fig.add_subplot(gs[1, 0])
pivot3 = df.pivot_table(values='B17：生活满意度', index='has_spouse', columns='health_poor', aggfunc='mean')
pivot3.index = ['无配偶','有配偶']; pivot3.columns = ['健康好','健康差']
sns.heatmap(pivot3, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax3,
            vmin=1.5, vmax=3.5, linewidths=0.5, annot_kws={'fontsize':10})
ax3.set_xlabel('自评健康', fontsize=11); ax3.set_ylabel('配偶状况', fontsize=11)
ax3.set_title('配偶×健康交互\n生活满意度均值', fontsize=11, fontweight='bold')

# (d) Age group x Internet
ax4 = fig.add_subplot(gs[1, 1])
df['age_group'] = pd.cut(df['年龄'], bins=[60,65,70,75,80,100], labels=['60-64','65-69','70-74','75-79','80+'])
pivot4 = df.pivot_table(values='B17：生活满意度', index='age_group', columns='internet', aggfunc='mean')
pivot4.columns = ['不上网','上网']
pivot4.plot(kind='bar', ax=ax4, color=['#e74c3c','#3498db'], edgecolor='white')
ax4.set_xlabel('年龄组', fontsize=11); ax4.set_ylabel('生活满意度均值', fontsize=11)
ax4.set_title('年龄组×上网状态的生活满意度', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9); ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

# (e) Social network x Health
ax5 = fig.add_subplot(gs[1, 2])
pivot5 = df.pivot_table(values='B17：生活满意度', index='D12_1：社交网络（家人/朋友）', columns='health_poor', aggfunc='mean')
pivot5.columns = ['健康好','健康差']
pivot5.index = ['0','1','2','3-4','5-8','9+'][:len(pivot5)]
pivot5.plot(kind='bar', ax=ax5, color=['#2ecc71','#e74c3c'], edgecolor='white')
ax5.set_xlabel('社交网络规模', fontsize=10); ax5.set_ylabel('生活满意度均值', fontsize=10)
ax5.set_title('社交网络×健康的生活满意度', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9); ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir,'fig7_multidim.png'),dpi=150,bbox_inches='tight'); plt.close()
print('Fig7 done')

# 保存结果
with open(os.path.join(_RES, 'analysis_ls_results.json'),'w',encoding='utf-8') as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=str)

print(f'\n=== 分析完成! 结果保存至 analysis_ls_results.json ===')
