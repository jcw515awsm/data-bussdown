# Data Bussdown

A lightweight data analysis project for exploring and modeling material volumes and defects.

## Setup

```bash
git clone https://github.com/yourusername/data-bussdown.git
cd data-bussdown

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Project Structure

```
data-bussdown/
├── data/                      # Input CSV data files
├── exploratory_analysis.py    # Initial EDA and visualization
├── simple_analysis.py         # Basic stats and summaries
├── glmm_analysis.py           # Generalized Linear Mixed Model analysis
├── ANALYSIS_SUMMARY.md        # Findings summary
├── STATISTICAL_RECOMMENDATIONS.md
├── requirements.txt
└── README.md
```

## Running Analyses

```bash
# Run exploratory data analysis
python exploratory_analysis.py

# Run GLMM model
python glmm_analysis.py

# Run basic summary statistics
python simple_analysis.py
```

## Updating Dependencies

```bash
pip freeze > requirements.txt
```

## Notes

- Use Python ≥ 3.10.
- Keep large datasets in `data/` (ignored in `.gitignore`).
- Avoid committing your virtual environment.
