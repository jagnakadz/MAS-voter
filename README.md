# Group 20: Voter Behviour Change Based on the Information Source

Tested with Python 3.13

Install: pip install -r requirements.txt
Run: python project-assignment-group-20.py

## Bimodal experiment

The bimodal experiment starts voters around two opinion peaks, 0.30 and 0.70, with a standard deviation of 0.10 within each group.

After installing the dependencies, run this command from the project folder:

```bash
python project-assignment-group-20.py bimodal
```

The experiment runs six conditions with 30 seeds each and saves the results to `results/bimodal/`. To compare party switching with the uniform start, use `results/bimodal/runs.csv` and `results/runs.csv`. Match runs by parameter values and seed.

