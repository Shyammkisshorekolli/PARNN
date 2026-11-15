from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='3. PARNN-Based Time Series Forecasting and Evaluation Platform', page_icon=str(APP_DIR / 'chart_icon.svg'), layout='wide')
st.markdown('''
<style>
:root { font-family: Arial, "Segoe UI", sans-serif; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: Arial, "Segoe UI", sans-serif; }
.stApp { background: #f3f5f7; color: #172b3a; }
.block-container { padding-top: 2.1rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid #aebfca !important; border-radius: 14px !important; box-shadow: 0 3px 12px rgba(23,43,58,.07); }
h1 { font-size: 2.35rem !important; line-height: 1.2 !important; font-weight: 700 !important; color: #172b3a; }
h2 { font-size: 1.8rem !important; line-height: 1.25 !important; font-weight: 700 !important; color: #172b3a; }
h3 { font-size: 1.42rem !important; line-height: 1.3 !important; font-weight: 650 !important; color: #172b3a; }
[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"], label { font-size: 1.05rem; line-height: 1.55; }
[data-testid="stCaptionContainer"] { color: #526577; }
[data-testid="stMetric"] { border: 1px solid #b7c8d2; border-left: 5px solid #1874b4; border-radius: 11px; padding: 15px; background: #fff; box-shadow: 0 3px 10px rgba(23,43,58,.08); }
[data-testid="stMetricLabel"] { font-size: 1rem !important; }
[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
[data-testid="stDataFrame"] { font-size: 1rem; }
.stButton button, .stDownloadButton button { font-size: 1rem; min-height: 2.8rem; padding: .55rem 1rem; }
.stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 650; }
[data-baseweb="select"] > div { background: #fff !important; border: 2px solid #9aafbc !important; border-radius: 9px !important; min-height: 2.85rem; box-shadow: 0 1px 4px rgba(23,43,58,.06); }
[data-baseweb="select"] > div:focus-within { border-color: #0878bd !important; box-shadow: 0 0 0 3px rgba(8,120,189,.22) !important; }
.dataset-ticker { overflow: hidden; white-space: nowrap; margin: .5rem 0 1.5rem; border: 1px solid #bfd0da; border-radius: 10px; background: #fff; box-shadow: 0 2px 8px rgba(23,43,58,.06); }
.dataset-ticker-track { display: inline-flex; gap: .7rem; padding: .7rem 0; width: max-content; animation: ticker 42s linear infinite; }
.dataset-ticker:hover .dataset-ticker-track { animation-play-state: paused; }
.dataset-chip { display: inline-block; border: 1px solid #cfdae1; border-radius: 999px; padding: .35rem .8rem; background: #f3f7fa; color: #19364a; font-size: .95rem; font-weight: 650; }
@keyframes ticker { from { transform: translateX(0); } to { transform: translateX(-50%); } }
@media (prefers-reduced-motion: reduce) { .dataset-ticker-track { animation: none; } }
</style>
''', unsafe_allow_html=True)

DATASETS = [
    'AMZN', 'Colombia_Dengue', 'Colombia_Malaria', 'GOOG', 'MSFT',
    'Russia_Exchange', 'Tourism', 'UK_Unemployment', 'US_EPU_Index',
    'Venezuela_Dengue', 'Venezuela_Malaria',
]
HORIZONS = ['ST', 'MT', 'LT']
NON_DEEP = ['PARNN', 'RWD', 'ETS', 'ARIMA', 'TBATS', 'ARNN', 'Hybrid-1', 'Hybrid-2']
DEEP = ['Hybrid-3', 'NBeats', 'DeepAR', 'TCN', 'Transformer']
ALL_MODELS = NON_DEEP + DEEP
METRICS = ['MASE', 'RMSE', 'SMAPE']
LABELS = {
    'AMZN': 'AMZN Stock', 'GOOG': 'GOOG Stock', 'MSFT': 'MSFT Stock',
    'Colombia_Dengue': 'Colombia Dengue', 'Colombia_Malaria': 'Colombia Malaria',
    'Venezuela_Dengue': 'Venezuela Dengue', 'Venezuela_Malaria': 'Venezuela Malaria',
    'Russia_Exchange': 'Russia Exchange', 'Tourism': 'Tourism',
    'UK_Unemployment': 'UK Unemployment', 'US_EPU_Index': 'US EPU Index',
}
DEFAULT_PROJECT = str(APP_DIR.parent if (APP_DIR.parent / 'results').exists() else APP_DIR)
st.title('3. PARNN-Based Time Series Forecasting and Evaluation Platform')
ticker_items = ''.join(f'<span class="dataset-chip">{LABELS.get(ds, ds)}</span>' for ds in DATASETS * 2)
st.markdown(f'<div class="dataset-ticker" aria-label="Selected datasets"><div class="dataset-ticker-track">{ticker_items}</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('Data source')
    project_text = st.text_input('Project folder', value=DEFAULT_PROJECT, help='Folder containing the results/ directory.')
    if st.button('Refresh saved results', use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()



@st.cache_data(ttl=45, show_spinner=False)
def load_results(project_text: str):
    root = Path(project_text) / 'results'
    files = []
    combined = root / 'repetition_scores.csv'
    if combined.exists():
        files.append(combined)
    else:
        for folder in ['parnn', 'baselines', 'deep']:
            path = root / folder
            if path.exists():
                files.extend(path.glob('*_reps.csv'))
    frames = []
    for file in files:
        try:
            d = pd.read_csv(file)
            if not {'Dataset', 'Horizon', 'Model', 'Rep', *METRICS}.issubset(d.columns):
                continue
            d = d[d['Dataset'].isin(DATASETS) & d['Horizon'].isin(HORIZONS)]
            d = d[d['Model'].isin(ALL_MODELS)]
            if not d.empty:
                frames.append(d[['Dataset', 'Horizon', 'Model', 'Rep', *METRICS]])
        except Exception:
            continue
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['Dataset','Horizon','Model','Rep',*METRICS])
    if not raw.empty:
        raw = raw.drop_duplicates(['Dataset','Horizon','Model','Rep'], keep='last')
        summary = raw.groupby(['Dataset','Horizon','Model'], as_index=False).agg(
            **{f'{m}_mean': (m, 'mean') for m in METRICS},
            **{f'{m}_sd': (m, 'std') for m in METRICS},
            n=('Rep', 'nunique'),
        )
    else:
        summary = pd.DataFrame(columns=['Dataset','Horizon','Model'] + [f'{m}_{stat}' for m in METRICS for stat in ['mean','sd']] + ['n'])
    return raw, summary, root

raw, summary, results_root = load_results(project_text)
forecast_path = APP_DIR / 'data' / 'parnn_forecast_plot_data.csv'
forecasts = pd.read_csv(forecast_path) if forecast_path.exists() else pd.DataFrame()
ext_path = APP_DIR / 'data' / 'extension_five_tasks.csv'
extension = pd.read_csv(ext_path) if ext_path.exists() else pd.DataFrame()

if not results_root.exists():
    st.error(f'Could not find results/ under: {project_text}. Change the project folder in the sidebar.')
    st.stop()
if raw.empty:
    st.warning('No selected-model repetition files were found. Check that the results folders contain *_reps.csv files.')
    st.stop()

# Task-level deep completion means all five selected deep models have at least 10 saved runs.
deep_tasks = []
for ds in DATASETS:
    for hz in HORIZONS:
        part = summary[(summary.Dataset == ds) & (summary.Horizon == hz)].set_index('Model')
        complete = all(m in part.index and int(part.loc[m, 'n']) >= 10 for m in DEEP)
        deep_tasks.append({'Dataset': ds, 'Horizon': hz, 'Deep_complete_10': complete})
task_status = pd.DataFrame(deep_tasks)

# Overview
complete_n = int(task_status.Deep_complete_10.sum())
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric('Selected datasets', len(DATASETS))
    c2.metric('Dataset–horizon tasks', len(DATASETS) * len(HORIZONS))
    c3.metric('Bundled forecast series', forecasts[['Dataset', 'Horizon']].drop_duplicates().shape[0] if not forecasts.empty else 0)

st.markdown('''This dashboard brings together the saved accuracy results, model rankings, PARNN forecasts, run coverage, and the available extension comparison. Use **Refresh saved results** after additional runs finish.''')

result_tab, rank_tab, forecast_tab, coverage_tab, extension_tab, thanks_tab = st.tabs([
    'Model results', 'Rankings', 'PARNN forecasts', 'Run coverage', 'Extension', 'Special Thanks',
])

with result_tab:
    st.subheader('Accuracy results')
    a, b = st.columns(2)
    ds = a.selectbox('Dataset', DATASETS, format_func=lambda x: LABELS.get(x, x), key='res_ds')
    hz = b.selectbox('Forecast horizon', HORIZONS, key='res_hz')
    view = summary[(summary.Dataset == ds) & (summary.Horizon == hz)].copy()
    view['Model group'] = np.where(view.Model.isin(DEEP), 'Deep / additional neural', 'Non-deep comparison')
    view = view[['Model','Model group','n'] + [f'{m}_mean' for m in METRICS] + [f'{m}_sd' for m in METRICS]]
    view = view.rename(columns={f'{m}_mean': f'{m} mean' for m in METRICS} | {f'{m}_sd': f'{m} SD' for m in METRICS})
    st.dataframe(view, hide_index=True, use_container_width=True, column_config={
        'n': st.column_config.NumberColumn('Runs', format='%d'),
        **{f'{m} mean': st.column_config.NumberColumn(f'{m} mean', format='%.4f') for m in METRICS},
        **{f'{m} SD': st.column_config.NumberColumn(f'{m} SD', format='%.4f') for m in METRICS},
    })
    csv = view.to_csv(index=False).encode('utf-8')
    st.download_button('Download this result table as CSV', csv, file_name=f'{ds}_{hz}_model_results.csv', mime='text/csv')

with rank_tab:
    st.subheader('Mean ranks by accuracy metric')
    a, b = st.columns(2)
    hz = a.selectbox('Forecast horizon', HORIZONS, key='rank_hz')
    model_set = b.radio('Model set', ['Non-deep models (8)', 'Deep-complete tasks (13 models)'], horizontal=True)
    metric = st.selectbox('Metric to rank', METRICS, key='rank_metric')
    models = NON_DEEP if model_set.startswith('Non-deep') else ALL_MODELS
    data = summary[(summary.Horizon == hz) & summary.Model.isin(models)].copy()
    if model_set.startswith('Deep-complete'):
        complete_keys = task_status[task_status.Deep_complete_10][['Dataset','Horizon']]
        data = data.merge(complete_keys, on=['Dataset','Horizon'], how='inner')
        data = data[data.Model.isin(ALL_MODELS)]
    pivot = data.pivot_table(index='Dataset', columns='Model', values=f'{metric}_mean', aggfunc='first').reindex(columns=models)
    pivot = pivot.dropna(axis=0, how='any')
    if pivot.empty:
        st.info('No complete dataset tasks are available for this model set and horizon yet.')
    else:
        per_dataset = pivot.rank(axis=1, method='average', ascending=True)
        mean_rank = per_dataset.mean(axis=0).sort_values()
        fig, ax = plt.subplots(figsize=(10, max(4, .48 * len(mean_rank))))
        color = ['#188f83' if m == 'PARNN' else '#53738f' for m in mean_rank.index]
        ax.barh(mean_rank.index, mean_rank.values, color=color, edgecolor='#34495e', linewidth=.5)
        ax.invert_yaxis(); ax.set_xlabel('Mean rank across complete dataset tasks (lower is better)')
        ax.set_title(f'{hz} · {metric} · {len(pivot)} tasks', fontweight='bold'); ax.grid(axis='x', alpha=.2)
        for i, val in enumerate(mean_rank.values): ax.text(val + .05, i, f'{val:.2f}', va='center', fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        rank_table = pd.DataFrame({'Model': mean_rank.index, 'Mean rank': mean_rank.values, 'Tasks': len(pivot)})
        st.dataframe(rank_table, hide_index=True, use_container_width=True)
    
with forecast_tab:
    st.subheader('PARNN forecasts against held-out observations')
    if forecasts.empty:
        st.warning('Bundled forecast plot data is unavailable.')
    else:
        a, b = st.columns(2)
        hz = a.selectbox('Forecast horizon', HORIZONS, key='fc_hz')
        ds = b.selectbox('Dataset', DATASETS, format_func=lambda x: LABELS.get(x, x), key='fc_ds')
        f = forecasts[(forecasts.Dataset == ds) & (forecasts.Horizon == hz)].sort_values('Index')
        if f.empty:
            st.info('No bundled PARNN forecast exists for this task.')
        else:
            start = int(f.ForecastStart.iloc[0])
            train, test = f[f.Index < start], f[f.Index >= start]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(train.Index, train.Value, color='#171717', lw=1, label='Input history')
            ax.plot(test.Index, test.Value, color='#1874b4', lw=1.7, label='Ground truth')
            ax.plot(test.Index, test.Forecast, color='#c62828', lw=1.7, label='PARNN forecast')
            ax.axvline(start - .5, color='#d98622', ls=':', lw=1.4, label='Forecast start')
            ax.set_title(f'{LABELS.get(ds,ds)} · {hz} horizon', fontweight='bold')
            ax.set_xlabel('Time index'); ax.set_ylabel('Value'); ax.grid(alpha=.18); ax.legend(ncol=4, frameon=True)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.download_button('Download forecast values as CSV', test[['Index','Value','Forecast']].rename(columns={'Value':'Actual'}).to_csv(index=False).encode('utf-8'), file_name=f'{ds}_{hz}_PARNN_forecast.csv', mime='text/csv')

with coverage_tab:
    st.subheader('Saved repetition coverage')
    coverage = summary.copy()
    coverage['Model group'] = np.where(coverage.Model.isin(DEEP), 'Deep / additional neural', 'Non-deep comparison')
    a, b, c = st.columns(3)
    hz = a.selectbox('Horizon', ['All horizons'] + HORIZONS, key='cov_hz')
    group = b.selectbox('Model group', ['All models', 'Non-deep comparison', 'Deep / additional neural'], key='cov_group')
    ds = c.selectbox('Dataset', ['All datasets'] + DATASETS, format_func=lambda x: LABELS.get(x, x), key='cov_ds')
    if hz != 'All horizons': coverage = coverage[coverage.Horizon == hz]
    if group != 'All models': coverage = coverage[coverage['Model group'] == group]
    if ds != 'All datasets': coverage = coverage[coverage.Dataset == ds]
    coverage = coverage[['Dataset','Horizon','Model','n']].sort_values(['Dataset','Horizon','Model'])
    st.dataframe(coverage, hide_index=True, use_container_width=True)
    st.download_button('Download coverage as CSV', coverage.to_csv(index=False).encode('utf-8'), file_name='run_coverage.csv', mime='text/csv')

with extension_tab:
    st.subheader('PARNN extension comparison')
    if extension.empty:
        st.info('No saved extension results are available.')
    else:
        st.markdown('Compare only dataset–horizon tasks and extension variants with saved results. The current extension experiment covers five tasks, each with 10 saved runs.')
        available_datasets = sorted(extension['Dataset'].dropna().unique().tolist())
        dataset = st.selectbox('Dataset with saved extension runs', available_datasets, key='ext_ds')
        task_rows = extension[extension.Dataset == dataset]
        available_horizons = sorted(task_rows['Horizon'].dropna().unique().tolist(), key=lambda x: HORIZONS.index(x) if x in HORIZONS else 99)
        horizon = st.selectbox('Horizon run for this dataset', available_horizons, key='ext_hz')
        task = task_rows[task_rows.Horizon == horizon].copy()
        variants = sorted(task['Extension'].dropna().unique().tolist())
        variant = st.selectbox('Extension variant', ['All tested variants'] + variants, key='ext_variant')
        best = task.loc[task['Extension_MASE'].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric('Original PARNN MASE', f"{float(best['PARNN_MASE']):.4f}", f"{int(best['PARNN_reps'])} saved runs")
        c2.metric('Best tested extension', str(best['Extension']), f"MASE {float(best['Extension_MASE']):.4f} · {int(best['Extension_reps'])} runs")
        c3.metric('Best MASE change', f"{float(best['Reduction_percent']):.1f}%", 'lower than original PARNN' if float(best['Reduction_percent']) >= 0 else 'higher than original PARNN')
        st.caption('Only saved task and variant combinations appear in the selectors. Results are observed values; missing runs are not filled or estimated.')
        display_task = task if variant == 'All tested variants' else task[task.Extension == variant]
        chart_data = display_task[['Extension','Extension_MASE']].copy()
        chart_data = pd.concat([pd.DataFrame([{'Extension':'Original PARNN','Extension_MASE':float(task['PARNN_MASE'].iloc[0])}]), chart_data], ignore_index=True)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        bar_colors = ['#55738e'] + ['#188f83' if m == best['Extension'] else '#87b8b1' for m in display_task['Extension']]
        bars = ax.bar(chart_data['Extension'], chart_data['Extension_MASE'], color=bar_colors, edgecolor='#34495e', linewidth=.5)
        ax.set_ylabel('MASE (lower is better)')
        ax.set_title(f'{LABELS.get(dataset,dataset)} · {horizon}: original vs saved extension results', fontweight='bold')
        ax.grid(axis='y', alpha=.2)
        ax.tick_params(axis='x', rotation=16)
        for bar, val in zip(bars, chart_data['Extension_MASE']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        table = display_task[['Dataset','Horizon','Extension','PARNN_MASE','PARNN_reps','Extension_MASE','Extension_reps','Reduction_percent','Best_extension_for_task']].copy()
        table = table.rename(columns={'PARNN_MASE':'Original PARNN MASE','PARNN_reps':'PARNN runs','Extension_MASE':'Extension MASE','Extension_reps':'Extension runs','Reduction_percent':'MASE reduction (%)','Best_extension_for_task':'Best for task'})
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={
            'Original PARNN MASE': st.column_config.NumberColumn(format='%.4f'),
            'PARNN runs': st.column_config.NumberColumn(format='%d'),
            'Extension MASE': st.column_config.NumberColumn(format='%.4f'),
            'Extension runs': st.column_config.NumberColumn(format='%d'),
            'MASE reduction (%)': st.column_config.NumberColumn(format='%.1f%%'),
            'Best for task': st.column_config.CheckboxColumn(),
        })
        st.download_button('Download the visible extension rows as CSV', table.to_csv(index=False).encode('utf-8'), file_name=f'{dataset}_{horizon}_extension_results.csv', mime='text/csv')




with thanks_tab:
    st.subheader('Special Thanks')
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #b7c8d2;border-left:6px solid #1874b4;
                    border-radius:14px;padding:1.5rem 1.7rem;box-shadow:0 4px 14px rgba(23,43,58,.08);">
          <p style="font-size:1.25rem;font-weight:700;color:#172b3a;margin-top:0;">
            With sincere gratitude to Rhythm Ma’am
          </p>
          <p>As someone who has always been interested in stock markets and trading, this course gave me a new way of looking at things. I used to mostly think about stocks in terms of their price movements, but learning about time-series forecasting made me realise how much more there is to explore through data, patterns, statistical models and neural networks.</p>
          <p>I am not yet sure how much of what we have learned I will be able to directly use in my future analysis, but I am certain that these concepts will come in handy in one way or another. I would love to explore them further and find some innovative ways to apply them, especially in understanding and analysing financial markets.</p>
          <p>I also want to thank you for bringing us to the front row. It might seem like a small thing, but it genuinely changed the way we listened to and experienced the class. It made the sessions feel more engaging and helped us connect better with what was being taught.</p>
          <p style="margin-bottom:0;">Thank you, Ma’am, for introducing us to this subject, encouraging us to think beyond what we already know, and making the learning experience memorable.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

