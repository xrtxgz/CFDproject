import streamlit as st
import pandas as pd
from FDFirst import CFDDiscovererWithFD
from datetime import datetime
from fd_utils import render_fd_tree, render_minfd_tree, render_minfd_list_with_deletion


st.set_page_config(page_title="CFD Discoverer", layout="wide")
st.title("CFD Discovery Tool (Graduation Project)")

# --- Sidebar: File Upload & Configuration ---
st.sidebar.header("📂 Upload & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

custom_header = st.sidebar.checkbox("Manually enter column names", False)

# --- Binning ---
n_bins_input = st.sidebar.text_input(
    "Discretization bins (optional)",
    placeholder="Leave blank for no binning",
    help="Enter an integer (e.g. 5) to discretize numeric columns, or leave blank for no binning"
)
n_bins = int(n_bins_input) if n_bins_input.strip().isdigit() else None

# --- Confidence ---
min_conf = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.5,
    max_value=1.0,
    value=0.95,
    step=0.01,
    help="Minimum confidence threshold (0.5 ~ 1.0)"
)

# --- Support ---
min_supp = st.sidebar.number_input(
    "Minimum Support (count or proportion)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.01,
    help="Enter a float <1 (as proportion) or integer ≥1 as row count"
)

# --- Max LHS size ---
maxsize = st.sidebar.number_input(
    "Max LHS attribute size",
    min_value=1,
    max_value=20,
    value=10,
    step=1,
    help="Maximum number of attributes allowed on the LHS"
)

# --- RHS index (optional) ---
rhs_index_input = st.sidebar.text_input(
    "RHS column index (optional)",
    placeholder="Leave blank for all",
    help="Enter column index (starting from 0) to limit RHS target"
)

# --- Top-K to display ---
topk = st.sidebar.number_input(
    "Top-K rules to show",
    min_value=5,
    max_value=50,
    value=20,
    step=1,
    help="How many top rules to display in result table"
)

# === Post-CFD repair parameter ===
repair_topk = st.sidebar.number_input(
    "Top-K vCFDs used for repair",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    help="Controls how many top variable CFDs are used for automatic repair"
)

# --- Confidence method ---
conf_method = st.sidebar.selectbox(
    "Confidence Calculation Method",
    ["overall", "avg", "min"],
    help="overall = global confidence; avg = average across groups; min = worst group confidence"
)

# --- Minimal FD direct search ---
direct_minfd = st.sidebar.checkbox("Use direct search for Minimal FD", value=True)

# --- Sidebar: Output Section Selection ---
st.sidebar.header("Select Output Sections")
output_options = st.sidebar.multiselect(
    "Choose what to display",
    ["FD", "Minimal FD", "CFD", "vCFD", "CFD Log", "vCFD Log"],
    default=["CFD", "vCFD"]
)

# --- Sidebar: Additional for vCFD ---
if "vCFD" in output_options or "vCFD Log" in output_options:
    allow_overlap = st.sidebar.checkbox("Allow Overlapping Variable CFD Patterns", False)
else:
    allow_overlap = False  # default fallback

# === Visualization ===
st.sidebar.header("FD Visualization")
rhs_vis = st.sidebar.text_input("RHS column name for FD visualization")
fd_vis = st.sidebar.button("Visualize FD Tree")
minfd_vis = st.sidebar.button("Visualize Minimal FD Tree")

# --- Main Logic ---
if uploaded_file:
    if custom_header:
        temp = pd.read_csv(uploaded_file, header=None)
        st.write(f"Detected {temp.shape[1]} columns")
        col_names = st.text_input("Enter comma-separated column names").split(",")
        if len(col_names) != temp.shape[1]:
            st.stop()
        df = pd.read_csv(uploaded_file, header=None, names=col_names)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    rhs_index = int(rhs_index_input) if rhs_index_input.isdigit() else None

    discoverer = CFDDiscovererWithFD(
        df,
        minconf=min_conf,
        maxsize=maxsize,
        rhs_index=rhs_index,
        conf_method=conf_method,
        n_bins=n_bins
    )

    # === Discovery & Display ===
    if "FD" in output_options:
        st.subheader("🔹 Functional Dependencies")
        fd_rules = discoverer.get_top_fds(topk=topk)
        fds_df = pd.DataFrame({"FD Rule": fd_rules})
        st.dataframe(fds_df)

        if st.checkbox("Export FD Rules as CSV"):
            csv_fd = fds_df.to_csv(index=False)
            today = datetime.now().strftime("%Y%m%d")
            st.download_button(
                "Download FD Rules",
                csv_fd,
                file_name=f"fds_{today}.csv",
                mime="text/csv"
            )

    if "Minimal FD" in output_options:
        st.subheader("🔹 Minimal Functional Dependencies")
        minfd_rules = discoverer.get_top_minimal_fds(topk=topk, direct=direct_minfd)
        minfd_df = pd.DataFrame({"minFD Rule": minfd_rules})
        st.dataframe(minfd_df)

        if st.checkbox("Export Minimal FD Rules as CSV"):
            csv_minfd = minfd_df.to_csv(index=False)
            today = datetime.now().strftime("%Y%m%d")
            st.download_button(
                "Download Minimal FD Rules",
                csv_minfd,
                file_name=f"minfds_{today}.csv",
                mime="text/csv"
            )

        # ✅ 添加这行代码，调用删除界面（基于当前列表）
        render_minfd_list_with_deletion(discoverer, df)

    if "CFD" in output_options:
        st.subheader("🔹 Constant CFDs")
        cfd_rules = discoverer.get_top_cfds(topk=topk, min_support=min_supp, rhs_index=rhs_index)
        st.info(cfd_rules[0])
        cfd_df = pd.DataFrame({"CFD Rule": cfd_rules[1:]})
        st.dataframe(cfd_df)

        if st.checkbox("Export CFD Rules as CSV"):
            csv_cfd = cfd_df.to_csv(index=False)
            today = datetime.now().strftime("%Y%m%d")
            st.download_button(
                "Download CFD Rules",
                csv_cfd,
                file_name=f"cfd_rules_{today}.csv",
                mime="text/csv"
            )

    if "vCFD" in output_options:
        st.subheader("🔹 Variable CFDs")
        vcfd_rules = discoverer.get_top_variable_cfds(
            topk=topk,
            min_support=min_supp,
            allow_overlap=allow_overlap,
            rhs_index=rhs_index
        )

        st.info(vcfd_rules[0])
        vcfd_df = pd.DataFrame({"Variable CFD Rule": vcfd_rules[1:]})
        st.dataframe(vcfd_df)

        if st.checkbox("Export vCFD Rules as CSV"):
            csv_vcfd = vcfd_df.to_csv(index=False)
            st.download_button(
                "Download vCFD Rules",
                csv_vcfd,
                file_name=f"vcfd_rules_{today}.csv",
                mime="text/csv"
            )

    if "CFD Log" in output_options:
        st.subheader("CFD Log")
        cfd_info, cfd_log = discoverer.discover_cfds_tracked(min_support=min_supp, rhs_index=rhs_index)
        st.json({**cfd_info, **cfd_log})

    if "vCFD Log" in output_options:
        st.subheader("Variable CFD Log")
        vcfds_info, vcfds_log = discoverer.discover_variable_cfds_tracked(min_support=min_supp,
                                                                          allow_overlap=allow_overlap, rhs_index=rhs_index)
        st.json({**vcfds_info, **vcfds_log})

    # === FD Tree Visualization ===
    if rhs_vis:
        if fd_vis:
            render_fd_tree(discoverer, rhs_vis, topk=topk)

        if minfd_vis:
            render_minfd_tree(discoverer, rhs_vis, topk=topk, direct_minfd=direct_minfd)

    # === Error detection & repair ===
    st.header("Post-CFD Options")
    if st.button("Detect Conflicts (vCFD violations)"):
        violations = discoverer.detect_cfd_violations(topk=repair_topk)
        st.write(f"Violated rows: {violations}")

    if st.button("🛠 Repair Data (based on vCFD)"):
        repaired_df = discoverer.repair_errors(topk=repair_topk)
        st.dataframe(repaired_df.head())
        discoverer.repaired_df = repaired_df
        differences = (df != repaired_df).sum().sum()
        changed_rows = (df != repaired_df).any(axis=1).sum()
        st.info(f"Total changed cells: {differences}")
        st.info(f"Total changed rows: {changed_rows}")
        st.subheader("Modified Rows (Preview)")
        diff_df = df[df != repaired_df].dropna(how="all")
        st.dataframe(diff_df.head())
        today = datetime.now().strftime("%Y%m%d")
        st.download_button(
            "Download Repaired Data",
            repaired_df.to_csv(index=False),
            file_name=f"repaired_data_{today}.csv"
        )


else:
    st.warning("📁 Please upload a CSV file to start.")
