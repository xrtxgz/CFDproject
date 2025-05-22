import streamlit as st
import pandas as pd

def build_minfd_df(discoverer):
    return pd.DataFrame([
        {
            "Index": i,
            "LHS": " AND ".join(lhs),
            "RHS": rhs,
            "Confidence": round(conf, 4)
        }
        for i, (lhs, rhs, conf) in enumerate(discoverer.minimal_fds, 1)
    ]) if discoverer.minimal_fds else pd.DataFrame()


def render_fd_tree(discoverer, rhs_vis: str, topk: int = 20):
    st.subheader(f"FD Tree: RHS = {rhs_vis}")
    discoverer.get_top_fds(topk=topk)
    discoverer.visualize_fd_candidates()

    fig = discoverer.repo.visualize_rhs_tree(rhs_vis)
    if fig:
        st.pyplot(fig)
    else:
        st.warning(f"No FD rules for RHS: {rhs_vis}")

def render_minfd_tree(discoverer, rhs_vis: str, topk: int = 20, direct_minfd: bool = True):
    st.subheader(f"Minimal FD Tree: RHS = {rhs_vis}")
    discoverer.get_top_minimal_fds(topk=topk, direct=direct_minfd)
    discoverer.visualize_minimal_fd_candidates()

    fig = discoverer.repo.visualize_rhs_tree(rhs_vis)
    if fig:
        st.pyplot(fig)
    else:
        st.warning("No Minimal FD tree available.")

def parse_index_input(index_str: str):
    indices = set()
    for part in index_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                indices.update(range(start, end + 1))
            except:
                pass
        elif part.isdigit():
            indices.add(int(part))
    return sorted(indices)

def render_minfd_list_with_deletion(discoverer, df):
    st.markdown("Delete Minimal FD Rules")

    def sync_minfd_df():
        st.session_state.minfd_df = pd.DataFrame([
            {
                "Index": i,
                "LHS": " AND ".join(lhs),
                "RHS": rhs,
                "Confidence": round(conf, 4)
            }
            for i, (lhs, rhs, conf) in enumerate(discoverer.minimal_fds, 1)
        ]) if discoverer.minimal_fds else pd.DataFrame()
        st.session_state.refresh_flags["Minimal FD"] = True

        # ✅ 添加以下内容
        st.session_state.refresh_flags["CFD"] = True
        st.session_state.refresh_flags["vCFD"] = True
        st.session_state.force_cfd_refresh = True
        st.session_state.pop("cfd_rules", None)
        st.session_state.pop("vcfd_rules", None)

    #Method One: Delete by column names
    with st.form("delete_minfd_lhs_rhs"):
        cols = df.columns.tolist()
        lhs_input = st.multiselect("Select LHS attributes", cols, key="minfd_lhs_del")
        rhs_input = st.selectbox("Select RHS attribute", cols, key="minfd_rhs_del")
        submit = st.form_submit_button("Delete by LHS → RHS")

    if submit:
        lhs = tuple(lhs_input)
        before = len(discoverer.minimal_fds)
        discoverer.minimal_fds = [r for r in discoverer.minimal_fds if not (r[0] == lhs and r[1] == rhs_input)]
        after = len(discoverer.minimal_fds)
        if after < before:
            st.success(f"Deleted rule: IF {' AND '.join(lhs)} THEN {rhs_input}")
            sync_minfd_df()
        else:
            st.warning("This rule does not exist.")

    #Method Two: Delete through index
    with st.expander("Delete by Index"):
        idx_str = st.text_input("Enter rule indices (e.g., 1,3-5)", key="minfd_index_del")
        if st.button("Delete by Index"):
            idx_list = parse_index_input(idx_str)
            before = len(discoverer.minimal_fds)
            discoverer.minimal_fds = [
                r for i, r in enumerate(discoverer.minimal_fds, 1) if i not in idx_list
            ]
            after = len(discoverer.minimal_fds)
            if after < before:
                st.success(f"Deleted {before - after} rule(s).")
                sync_minfd_df()
            else:
                st.warning("No matching rule index.")


