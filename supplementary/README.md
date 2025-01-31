# Neural Heaviside SDF Topology Optimizer

<p align="center">
  <figure style="display: inline-block; width: 25%;">
    <img src="src/Brecket_fm_ae_hv_ellipse_final.png" alt="Brecket_fm_ae_hv_ellipse_final">
    <figcaption>Brecket FM Ellipses Solution</figcaption>
  </figure>
  <figure style="display: inline-block; width: 25%;">
    <img src="src/Brecket_fm_ae_hv_nnHv_final.png" alt="Brecket_fm_ae_hv_nnHv_final">
    <figcaption>Brecket FM NHvSDF Solution</figcaption>
  </figure>
  <figure style="display: inline-block; width: 25%;">
    <img src="src/Brecket_SIMP_final.png" alt="Brecket_SIMP_final">
    <figcaption>Brecket SIMP Solution</figcaption>
  </figure>
</p>
The Neural Heaviside SDF Topology Optimizer project focuses on Feature-mapping Topology Optimization based on Neural Signed Distance Functions (Neural SDF).

## Generating Datasets

To generate datasets, use the following command or refer to the notebook examples in the `dataset_generation` folder. It is recommended not to change the `root_path`, as it is used in other scripts.

```bash
cd dataset_generation
python generate_datasets.py --root_path ../shape_datasets
```

## Training Models

To train the models, use the scripts in the `scripts` folder.

### First Strategy

Run the following command from the root folder:

```bash
bash scripts/train_first_strategy.sh
```

### Second Strategy

Run the following command from the root folder:

```bash
bash scripts/train_second_strategy.sh
```

### Comparison Table

To generate the comparison table, run the following command from the root folder:

```bash
python scripts/comparison_table.py --strategy first
```

## Saving Z Limits

Before running the Feature-mapping Topology Optimization with the trained models, you need to save the z limits of the models.

```bash
bash scripts/investigate_z_limits.sh
```

## Notebook Examples

Refer to the notebook examples in the `NN_TopOpt` folder for more examples:

- **SIMP Topology Optimization:** `SIMP_examples.ipynb`
- **Feature-mapping Topology Optimization with Ellipses:** `Ellipses_examples.ipynb`
- **Topology Optimization with Neural Heaviside SDF:** `NHSDF_examples.ipynb`

To generate new tasks, see the notebook `test_problems/create_problem.ipynb`.
