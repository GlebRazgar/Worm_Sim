# Worm Sim 🪱 
**Simulating C. elegans from first principles**

A comprehensive simulation framework for modeling the brain, body, and environment interactions of *Caenorhabditis elegans* using integrative data-driven approaches.

![C. elegans crawling simulation](assets/worm-crawling.gif)

## 🎯 Overview

This project implements a multi-scale simulation of C. elegans, incorporating:
- **🧠 Neural Connectome**: Structural and functional brain modeling
- **🦴 Body Physics**: MuJoCo-based biomechanical simulation  
- **🌍 Environment**: Interactive environmental responses
- **🔬 Data Integration**: Real connectome data and experimental validation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Conda package manager
- MuJoCo physics engine

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Worm_Sim.git
cd Worm_Sim
```

2. **Set up the environment:**
```bash
conda env create -f environment.yml
conda activate worm_sim
```

## 📊 Data Visualization

### Connectome Viewer
Explore the 3D neural connectome structure:
```bash
open -a Safari c_elegans_connectome.html
```

### Tissue Data Viewer
Launch the Neuroglancer server for high-resolution data exploration:
```bash
python data/view_data.py
```
Then access via: `http://localhost:YOUR-IP/v/NEUROGLANER-PATTERN`

## 🎮 Simulation

### Body Physics Simulation
Run the MuJoCo-based biomechanical simulation:
```bash
mjpython body_sim.py
```

### Neural Network Simulation
Execute the spiking neural network model:
```bash
python snn_connectome.py
```

## 📁 Project Structure

```
Worm_Sim/
├── assets/                 # Visual assets and animations
├── data/                   # Raw and processed data
├── first_principles_worm/  # Core simulation modules
├── physics_sim/           # MuJoCo physics simulations
├── outputs/               # Simulation results
└── neuroml/              # NeuroML model definitions
```

## 🔬 Research Context

This simulation framework is inspired by recent advances in C. elegans connectomics and aims to replicate findings from integrative brain-body-environment modeling studies. The project incorporates:

- **Structural Connectome Data**: Based on White et al. (1986) and subsequent refinements
- **Functional Validation**: Calcium imaging and electrophysiology data integration
- **Biomechanical Modeling**: Realistic body dynamics and locomotion patterns

---

*Built with ❤️ from Duality for the future of humanity
