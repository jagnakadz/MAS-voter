# Group 20: Voter Behaviour Change Based on the Information Source

Requires Python 3.12 or newer (tested with Python 3.13).

```bash
pip install -r requirements.txt                              # install
python project-assignment-group-20.py test                   # run the tests
python project-assignment-group-20.py                        # full experiment (25+ minutes)
solara run project-assignment-group-20-visualisation.py      # interactive visualisation
```

Run everything from the main project folder. The full experiment overwrites the files in `results/`.

## Folders

- `results/`: experiment data
- `results/analysis/`: analysis scripts (see the README there)
- `verification/`: tests, reference model and sensitivity analysis

## Bimodal experiment

The bimodal experiment starts voters around two opinion peaks, 0.30 and 0.70, with a standard deviation of 0.10 within each group.

After installing the dependencies, run this command from the project folder:

```bash
python project-assignment-group-20.py bimodal
```

The experiment runs six conditions with 30 seeds each and saves the results to `results/bimodal/`. To compare party switching with the uniform start, use `results/bimodal/runs.csv` and `results/runs.csv`. Compare conditions by parameter values. Runs with the same seed are not paired between the two starts.

