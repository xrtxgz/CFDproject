import streamlit as st
import pandas as pd

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
    st.markdown("### 📋 Current Minimal FD Rules (Editable)")

    if not discoverer.minimal_fds:
        st.info("No Minimal FDs available.")
        return

    rule_data = []
    for i, (lhs, rhs, conf) in enumerate(discoverer.minimal_fds, 1):
        lhs_str = " AND ".join(lhs)
        rule_data.append({"Index": i, "LHS": lhs_str, "RHS": rhs, "Confidence": round(conf, 4)})

    df_rules = pd.DataFrame(rule_data)
    st.dataframe(df_rules, use_container_width=True)

    st.markdown("#### 🧹 Delete Minimal FD Rules")

    # 方法一：通过列名删除
    with st.form("delete_minfd_lhs_rhs"):
        cols = df.columns.tolist()
        lhs_input = st.multiselect("Select LHS attributes", cols, key="minfd_lhs_del")
        rhs_input = st.selectbox("Select RHS attribute", cols, key="minfd_rhs_del")
        submit_1 = st.form_submit_button("Delete by LHS → RHS")

    if submit_1:
        lhs = tuple(lhs_input)
        match = [r for r in discoverer.minimal_fds if r[0] == lhs and r[1] == rhs_input]
        if match:
            discoverer.minimal_fds = [r for r in discoverer.minimal_fds if not (r[0] == lhs and r[1] == rhs_input)]
            st.success(f"Deleted rule: IF {' AND '.join(lhs)} THEN {rhs_input}")
        else:
            st.warning("This rule does not exist.")

    # 方法二：通过索引批量删除
    with st.expander("Delete by Index"):
        idx_str = st.text_input("Enter rule indices (e.g., 1,3-5)", key="minfd_index_del")
        if st.button("Delete by Index"):
            try:
                idx_list = parse_index_input(idx_str)
                all_rules = discoverer.minimal_fds
                filtered = []
                for i, rule in enumerate(all_rules, 1):
                    if i not in idx_list:
                        filtered.append(rule)
                deleted = len(all_rules) - len(filtered)
                discoverer.minimal_fds = filtered
                st.success(f"Deleted {deleted} rule(s) by index.")
            except Exception as e:
                st.error(f"Failed to delete: {e}")

    # 更新展示
    if discoverer.minimal_fds:
        updated_data = []
        for i, (lhs, rhs, conf) in enumerate(discoverer.minimal_fds, 1):
            lhs_str = " AND ".join(lhs)
            updated_data.append({"Index": i, "LHS": lhs_str, "RHS": rhs, "Confidence": round(conf, 4)})
        st.markdown("### 🔁 Updated Minimal FDs")
        st.dataframe(pd.DataFrame(updated_data), use_container_width=True)
    else:
        st.warning("All minimal FD rules have been deleted.")

