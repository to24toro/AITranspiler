use ndarray::{Array2, Array3, Axis};
use rand::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::HashSet;

#[pymodule]
fn game_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Game>()?;
    Ok(())
}

#[pyclass]
pub struct GameSettings {
    #[pyo3(get, set)]
    pub gate: f64,
    #[pyo3(get, set)]
    pub layer: f64,
    #[pyo3(get, set)]
    pub max_steps: usize,
}

#[pymethods]
impl GameSettings {
    #[new]
    pub fn new(gate: f64, layer: f64, max_steps: usize) -> Self {
        GameSettings {
            gate,
            layer,
            max_steps,
        }
    }
}

#[pyclass]
pub struct Game {
    qubits: usize,
    gate: f64,
    layer_penalty: f64,
    max_steps: usize,
    coupling_map: Vec<(usize, usize)>,
    coupling_map_mat: Array2<f64>,
    used_columns_set: HashSet<usize>,
    current_layer: usize,
    pub action_space: usize,
}

#[pymethods]
impl Game {
    #[new]
    pub fn new(qubits: usize, settings: GameSettings) -> Self {
        let coupling_map = Self::generate_coupling_map(qubits);
        let coupling_map_mat = Self::generate_coupling_map_mat(qubits, &coupling_map);
        let action_space = coupling_map.len();

        Game {
            qubits,
            gate: settings.gate,
            layer_penalty: settings.layer,
            max_steps: settings.max_steps,
            coupling_map,
            coupling_map_mat,
            used_columns_set: std::collections::HashSet::new(),
            current_layer: 1,
            action_space,
        }
    }

    pub fn get_initial_state(&self) -> Array2<f64> {
        // Pythonと同様の行列生成ロジック
        let mut rng = thread_rng();
        let mut upper_triangle = Array2::<f64>::zeros((self.qubits, self.qubits));
        for i in 0..self.qubits {
            for j in i + 1..self.qubits {
                upper_triangle[[i, j]] = rng.gen_range(0..2) as f64;
            }
        }
        let mut symmetric_matrix = upper_triangle.clone();
        for i in 0..self.qubits {
            for j in 0..i {
                symmetric_matrix[[i, j]] = symmetric_matrix[[j, i]];
            }
        }
        for i in 0..self.qubits {
            symmetric_matrix[[i, i]] = 0.0;
        }
        &symmetric_matrix - &( &symmetric_matrix * &self.coupling_map_mat )
    }

    pub fn get_valid_actions(&self, py: Python, state: &PyTuple) -> Vec<usize> {
        let state: Array2<f64> = state.extract().unwrap();
        (0..self.action_space)
            .filter(|&action| self.is_valid_action(&state, action, None))
            .collect()
    }

    pub fn generate_coupling_map(qubits: usize) -> Vec<(usize, usize)> {
        (0..qubits - 1).map(|i| (i, i+1)).collect()
    }
}
