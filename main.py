import streamlit as st
import os
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

# --- Column Exclude ---
exclude_cols_str = st.sidebar.text_input(
    "Column indices to exclude (comma-separated)",
    placeholder="e.g., 1,3,5",
    help="Enter column indices (starting from 0) to exclude from analysis"
)

# --- Built-in Datasets ---
st.sidebar.markdown("---")
st.sidebar.header("Built-in Test Datasets")

builtin_dir = "DataSet"
available_files = [f for f in os.listdir(builtin_dir) if f.endswith(".csv")]

builtin_choice = st.sidebar.selectbox(
    "Choose a test dataset",
    ["None"] + available_files
)

use_builtin = builtin_choice != "None"

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
sections = ["FD", "Minimal FD", "CFD", "vCFD", "CFD Log", "vCFD Log"]
if "visible_sections" not in st.session_state:
    st.session_state.visible_sections = {sec: False for sec in sections}
if "refresh_flags" not in st.session_state:
    st.session_state.refresh_flags = {sec: False for sec in sections}
if "repair_triggered" not in st.session_state:
    st.session_state.repair_triggered = False
if "violation_triggered" not in st.session_state:
    st.session_state.violation_triggered = False
if "violated_rows" not in st.session_state:
    st.session_state.violated_rows = []


st.sidebar.header("Select Output Section")
for sec in sections:
    if st.sidebar.button(sec):
        st.session_state.visible_sections[sec] = True
        st.session_state.refresh_flags[sec] = True  # 标记该区域需要刷新

# --- Sidebar: Additional for vCFD ---
if any([
    st.session_state.visible_sections.get("vCFD", False),
    st.session_state.visible_sections.get("vCFD Log", False)
]):
    allow_overlap = st.sidebar.checkbox("Allow Overlapping Variable CFD Patterns", False)
else:
    allow_overlap = False # default fallback

# === Visualization ===
st.sidebar.header("FD Visualization")
rhs_vis = st.sidebar.text_input("RHS column name for FD visualization")
fd_vis = st.sidebar.button("Visualize FD Tree")
minfd_vis = st.sidebar.button("Visualize Minimal FD Tree")

# --- Main Logic ---
df = None

# --- Parse column exclusion input ---
def parse_index_list(s):
    try:
        return sorted(set(int(i.strip()) for i in s.split(",") if i.strip().isdigit()))
    except:
        return []

excluded_indices = parse_index_list(exclude_cols_str)

if use_builtin:
    df_path = os.path.join(builtin_dir, builtin_choice)
    df = pd.read_csv(df_path)
    st.success(f"Loaded built-in dataset: {builtin_choice}")
elif uploaded_file:
    if custom_header:
        temp = pd.read_csv(uploaded_file, header=None)
        st.write(f"Detected {temp.shape[1]} columns")
        col_names = st.text_input("Enter comma-separated column names").split(",")
        if len(col_names) != temp.shape[1]:
            st.stop()
        df = pd.read_csv(uploaded_file, header=None, names=col_names)
    else:
        df = pd.read_csv(uploaded_file)

if 'df' in locals() and excluded_indices:
    if max(excluded_indices) < df.shape[1]:
        df.drop(df.columns[excluded_indices], axis=1, inplace=True)
    else:
        st.warning("Some excluded indices exceed column count. Ignored.")

if df is not None:
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
    discoverer.get_top_minimal_fds(topk=topk, direct=direct_minfd)

    # --- Display each section if visible ---
    if st.session_state.visible_sections.get("FD"):
        st.subheader("🔹 Functional Dependencies")
        if st.session_state.refresh_flags["FD"]:
            st.session_state.fd_df = pd.DataFrame({"FD Rule": discoverer.get_top_fds(topk=topk)})
            st.session_state.refresh_flags["FD"] = False
        st.dataframe(st.session_state.fd_df)
        if st.checkbox("Export FD Rules as CSV"):
            csv_fd = st.session_state.fd_df.to_csv(index=False)
            today = datetime.now().strftime("%Y%m%d")
            st.download_button(
                "Download FD Rules",
                csv_fd,
                file_name=f"fds_{today}.csv",
                mime="text/csv"
            )

    if st.session_state.visible_sections.get("Minimal FD"):
        st.subheader("🔹 Minimal Functional Dependencies")

        if "minfd_df" not in st.session_state or st.session_state.refresh_flags.get("Minimal FD", False):
            discoverer.get_top_minimal_fds(topk=topk, direct=False)

            st.session_state.minfd_df = pd.DataFrame([
                {
                    "Index": i,
                    "LHS": " AND ".join(lhs),
                    "RHS": rhs,
                    "Confidence": round(conf, 4)
                }
                for i, (lhs, rhs, conf) in enumerate(discoverer.minimal_fds, 1)
            ]) if discoverer.minimal_fds else pd.DataFrame()

            st.session_state.refresh_flags["Minimal FD"] = False

        if not st.session_state.minfd_df.empty:
            st.dataframe(st.session_state.minfd_df)
        else:
            st.info("No Minimal FDs available.")

        render_minfd_list_with_deletion(discoverer, df)

        if st.button("Apply Updated Minimal FDs"):
            discoverer.discover_minimal_fds(direct=direct_minfd)
            st.session_state.refresh_flags["Minimal FD"] = True
            st.session_state.refresh_flags["CFD"] = True
            st.session_state.refresh_flags["vCFD"] = True
            st.session_state.force_cfd_refresh = True

    if st.session_state.visible_sections.get("CFD"):
        st.subheader("🔹 Constant CFDs")
        if st.session_state.refresh_flags["CFD"]:
            rules = discoverer.get_top_cfds(
                topk=topk,
                min_support=min_supp,
                rhs_index=rhs_index,
                force=st.session_state.get("force_cfd_refresh", False)
            )
            st.session_state.cfd_info = rules[0]
            st.session_state.cfd_df = pd.DataFrame({"CFD Rule": rules[1:]})
            st.session_state.refresh_flags["CFD"] = False
            st.session_state.force_cfd_refresh = False
        st.info(st.session_state.cfd_info)
        st.dataframe(st.session_state.cfd_df)
        if st.checkbox("Export CFD Rules as CSV"):
                csv_cfd = st.session_state.cfd_df.to_csv(index=False)
                today = datetime.now().strftime("%Y%m%d")
                st.download_button(
                    "Download CFD Rules",
                    csv_cfd,
                    file_name=f"cfd_rules_{today}.csv",
                    mime="text/csv"
                )

    if st.session_state.visible_sections.get("vCFD"):
        st.subheader("🔹 Variable CFDs")
        if st.session_state.refresh_flags["vCFD"]:
            rules = discoverer.get_top_variable_cfds(
                topk=topk,
                min_support=min_supp,
                allow_overlap=allow_overlap,
                rhs_index=rhs_index,
                force=st.session_state.get("force_cfd_refresh", False)
            )
            st.session_state.vcfd_info = rules[0]
            st.session_state.vcfd_df = pd.DataFrame({"Variable CFD Rule": rules[1:]})
            st.session_state.refresh_flags["vCFD"] = False
            st.session_state.force_cfd_refresh = False
        st.info(st.session_state.vcfd_info)
        st.dataframe(st.session_state.vcfd_df)
        if st.checkbox("Export vCFD Rules as CSV"):
                csv_vcfd = st.session_state.vcfd_df.to_csv(index=False)
                today = datetime.now().strftime("%Y%m%d")
                st.download_button(
                    "Download vCFD Rules",
                    csv_vcfd,
                    file_name=f"vcfd_rules_{today}.csv",
                    mime="text/csv"
                )

    if st.session_state.visible_sections.get("CFD Log"):
        st.subheader("CFD Log")
        if st.session_state.refresh_flags["CFD Log"]:
            info, log = discoverer.discover_cfds_tracked(min_support=min_supp, rhs_index=rhs_index)
            st.session_state.cfdlog = {**info, **log}
            st.session_state.refresh_flags["CFD Log"] = False
        st.json(st.session_state.cfdlog)


    if st.session_state.visible_sections.get("vCFD Log"):
        st.subheader("Variable CFD Log")
        if st.session_state.refresh_flags["vCFD Log"]:
            info, log = discoverer.discover_variable_cfds_tracked(min_support=min_supp, allow_overlap=allow_overlap,
                                                                  rhs_index=rhs_index)
            st.session_state.vcfdlog = {**info, **log}
            st.session_state.refresh_flags["vCFD Log"] = False
        st.json(st.session_state.vcfdlog)

    # === FD Tree Visualization ===
    if rhs_vis:
        if fd_vis:
            render_fd_tree(discoverer, rhs_vis, topk=topk)
        if minfd_vis:
            render_minfd_tree(discoverer, rhs_vis, topk=topk, direct_minfd=direct_minfd)

    # === Error detection & repair ===
    st.header("Post-CFD Options")
    if st.button("Detect Conflicts (vCFD violations)"):
        st.session_state.violated_rows = discoverer.detect_cfd_violations(
            topk=repair_topk,
            min_support=min_supp,
            allow_overlap=allow_overlap,
            rhs_index=rhs_index
        )
        st.session_state.violation_triggered = True
    if st.session_state.violation_triggered:
        violations = st.session_state.violated_rows
        st.success(f"Total of {len(violations)} violation records were detected")
        st.code(", ".join(map(str, violations[:50])) + (" ..." if len(violations) > 50 else ""))

    if st.button("🛠 Repair Data (based on vCFD)"):
        if not st.session_state.violated_rows:
            st.warning("Please click 'Detect Conflicts' first to detect violations")
        else:
            repaired_df = discoverer.repair_errors(
                topk=repair_topk,
                min_support=min_supp,
                allow_overlap=allow_overlap,
                rhs_index=rhs_index
            )
            discoverer.repaired_df = repaired_df

            df_str = df.astype(str)
            repaired_str = repaired_df.astype(str)
            diff_mask = df_str != repaired_str
            differences = diff_mask.sum().sum()
            changed_rows = diff_mask.any(axis=1).sum()

        st.success(f"Total check {len(df)} rows，discover {changed_rows} rows have changed，total {differences} cells have changed")

        modified_preview = repaired_df[diff_mask.any(axis=1)].copy().head(10)
        st.subheader("Modified Rows (Preview, up to 10):")
        if not modified_preview.empty:
            st.dataframe(modified_preview)
        else:
            st.info("No records that need to be repaired were detected.")

        today = datetime.now().strftime("%Y%m%d")
        st.download_button(
            "Download Repaired Data",
            repaired_df.to_csv(index=False),
            file_name=f"repaired_data_{today}.csv"
        )

else:
    st.warning("📁 Please upload a CSV file to start.")
