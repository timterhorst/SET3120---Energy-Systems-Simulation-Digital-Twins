# SET3120 - Energy Systems Simulation & Digital Twins

Master's level course project at TU Delft implementing a Smart House in a Smart Neighborhood using Python-based co-simulation.

## Project Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── configurations/     # YAML configuration files
├── data/              # CSV input files
├── src/               # Python modules
└── Practicums/        # Practicum exercises
```

## Expected Python Modules (Base Implementation - Week 5)

- `controller.py` - Heat pump controller logic
- `cosim_framework.py` - Model and Manager classes
- `grid.py` - PowerGridModel wrapper for electricity grid
- `heat_pump.py` - Heat pump thermal model
- `room.py` - Room thermal ODE model
- `load_configurations.py` - YAML configuration loader
- `run_co_simulation.py` - Main execution script

## Dependencies

- `power-grid-model` - For electrical power flow calculations
- `pandas` - Data handling
- `numpy` - Numerical operations
- `matplotlib` - Plotting results
- `pyyaml` - Configuration file handling

## Branches

- `practicum-work` - Current practicum exercises and work
- `project` - Base implementation (to be provided in Week 5)

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run practicum code from `practicum-work` branch
4. Base implementation will be merged into `project` branch in Week 5
