# PARNN-Based Time Series Forecasting and Evaluation Platform

Interactive Streamlit dashboard for exploring saved PARNN forecasting results. The deployment package includes the observed repetition-level scores, selected PARNN forecast series, and the recorded five-task extension comparison. It does not generate or fill in missing scores.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud and connect your GitHub account.
2. Choose repository `Shyammkisshorekolli/PARNN`, branch `main`, and file `app.py`.
3. Select **Deploy**. Streamlit installs the packages listed in `requirements.txt`.

## Dashboard sections

- **Model results:** observed MASE, RMSE, and SMAPE summaries and repetition counts.
- **Rankings:** dataset-level average ranks for each metric and forecast horizon.
- **PARNN forecasts:** bundled input history, held-out observations, and PARNN forecasts.
- **Run coverage:** saved repetition counts by dataset, horizon, and model.
- **Extension:** saved MASE comparisons for the five completed dataset–horizon tasks.
- **Special Thanks:** acknowledgement to Rhythm Ma’am.

The app excludes Births and MLP from this selected view. Missing scores remain missing.

