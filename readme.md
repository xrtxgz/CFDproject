# CFD: FD-First Algorithm Tool (Graduation Project)

This is an interactive tool based on Streamlit, designed to discover, visualize, and repair various dependency rules from relational datasets:

- Functional Dependencies (FD)
- Minimal Functional Dependencies (Minimal FD)
- Constant Conditional Functional Dependencies (CFD)
- Variable Conditional Functional Dependencies (Variable CFD / vCFD)

## Features

- Upload your own CSV dataset, or select sample datasets from the `DataSet` folder
- Supports discovery of:
  - All FD rules
  - Minimal FDs (with support for deletion, updating, and direct minimal FD mining)
  - Constant CFDs (with support and confidence thresholds)
  - Variable CFDs (with pattern generalization and overlap control)
- Visualize FD and Minimal FD as tree structures (grouped by RHS)
- Configure key parameters: support, confidence, max LHS size, target RHS, Top-K output, and discretization bins
- View discovery logs for CFD processes
- Perform error detection and automatic repair
- Export discovered rules and repaired data as CSV files

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit app

```bash
streamlit run main.py
```

### 3. Upload dataset or use built-in samples

- Format: CSV file
- Optional settings:
  - Manually specify column names
  - Exclude columns by their index

You can also select built-in datasets located in the `DataSet/` directory.

---

## Parameter Description

| Parameter           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| Minimum Support     | Accepts a float (e.g., 0.05 for 5%) or an integer (e.g., 10 means ≥10 records) |
| Minimum Confidence  | Range: 0.5 to 1.0. Higher means stricter rule reliability                   |
| Max LHS Size        | Maximum number of attributes on the left-hand side of the rule              |
| RHS Column Index    | Optionally fix the right-hand side to a specific column index               |
| Top-K Rules         | Display the top K rules ranked by support or confidence                     |
| Discretization Bins | Optional bin count for numerical columns (e.g., 4 splits values into 4 ranges) |

---

## Project Structure

.
├── main.py # Streamlit frontend and UI logic
├── FDFirst.py # Core discovery of FD, Minimal FD, CFD, and vCFD
├── fd_utils.py # Visualization and interactive logic for FD tree & lists
├── pattern_trie.py # Prefix tree structure for vCFD overlap control
├── rule_repository.py # Rule storage, management, and repair functionality
├── confidence.py # Confidence calculation for CFDs
├── DataSet/ # Sample datasets for testing

---

## Output Files

After running the tool, the following outputs may be generated:

- **Tree visualizations** for FD and Minimal FD (grouped by RHS)
- **Tabular rule displays** for:
  - Functional Dependencies
  - Minimal Functional Dependencies
  - Constant CFDs
  - Variable CFDs
- **Exportable CSVs**:
  - `fd_rules.csv` – Full FD list
  - `minimal_fd_rules.csv` – Filtered minimal FD list
  - `cfds.csv` – Constant CFDs (meet support/confidence thresholds)
  - `vcfds.csv` – Variable CFDs (may include overlapping/generalized patterns)
  - `repaired_data.csv` – Dataset after vCFD-based repairs
  - `discovery_log.csv` – Time/memory/rule statistics from CFD discovery

---

## Environment Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

requirements.txt includes:

```shell
streamlit>=1.20.0
pandas>=1.4.0
numpy>=1.22.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
networkx>=2.6
```