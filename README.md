# Worm Sim 🪱 
Simulating C.elegans from first principles

### Installation
Requirements installation:
```
conda env create -f environment.yml
```

### Dataviewing
CONNECTOME VIEWING:
To view the connectome rendered in 3D run the following in terminal:
```open -a Safari c_elegans_connectome.html```

TISSUE VIEWING:
To view data & labels launch the neuroglancer server by running ```python data/view_data.py```
Then locally view it in your browser using this pattern:
http://localhost:YOUR-IP/v/NEUROGLANER-PATTERN

### Simulation
Running MuJoCo through terminal:
```mjpython body_sim.py```