import streamlit as st
import pandas as pd
from FDFirst import CFDDiscovererWithFD

st.title("🎓 CFD Discoverer 毕设项目")

uploaded_file = st.file_uploader("上传数据集 CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("数据预览：", df.head())

    cfd = CFDDiscovererWithFD(df)
    results = cfd.discover_cfds()
    st.write("发现的 CFD：")
    st.dataframe(results)
