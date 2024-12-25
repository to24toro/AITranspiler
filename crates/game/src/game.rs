use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use ndarray::{Array2, ArrayBase, DataMut, Ix2};
use std::collections::HashSet;
use rand::Rng;

#[derive(Debug)]
pub struct GameSettings {
    pub gate: f64,
    pub layer: f64,
    pub max_steps: usize,
}


pub struct Game {
    pub qubits: usize,
    pub gate: f64,
    pub layer_penalty: f64,
    pub max_steps: usize,
    pub coupling_map: Vec<(usize, usize)>,
    pub coupling_map_mat: Array2<f64>,
    pub used_columns_set: HashSet<usize>,
    pub current_layer: usize,
    pub action_space: usize,
}

impl Game {
    pub fn new(qubits: usize, config: &GameSettings) -> Self {
        let gate = config.gate;
        let layer_penalty = config.layer;
        let max_steps = config.max_steps;

        let coupling_map = Self::generate_coupling_map(qubits);
        let coupling_map_mat = Self::generate_coupling_map_mat(qubits, &coupling_map);

        let action_space = coupling_map.len();
        Game {
            qubits,
            gate,
            layer_penalty,
            max_steps,
            coupling_map,
            coupling_map_mat,
            used_columns_set: HashSet::new(),
            current_layer: 1,
            action_space,
        }
    }

    fn generate_coupling_map(qubits: usize) -> Vec<(usize, usize)> {
        (0..(qubits - 1))
            .map(|i| (i, i + 1))
            .collect()
    }

    fn generate_coupling_map_mat(
        qubits: usize,
        coupling_map: &[(usize, usize)],
    ) -> Array2<f64> {
        let mut mat = Array2::<f64>::zeros((qubits, qubits));
        for &(i, j) in coupling_map {
            mat[[i, j]] = 1.0;
            mat[[j, i]] = 1.0;
        }
        mat
    }

    pub fn reset_used_columns(&mut self) {
        self.used_columns_set.clear();
        self.current_layer += 1;
    }

    pub fn is_done(&self, mat: &Array2<f64>) -> bool {
        mat.iter().all(|&val| val.abs() < f64::EPSILON)
    }

    fn is_valid_action(
        &self,
        mat: &Array2<f64>,
        action: usize,
        prev_action: Option<usize>,
    ) -> bool {

        let (mut col1, mut col2) = self.coupling_map[action];
        if col1 > col2 {
            std::mem::swap(&mut col1, &mut col2);
        }

        let col1_all_zero = mat.column(col1).iter().all(|&v| v.abs() < f64::EPSILON);
        let col2_all_zero = mat.column(col2).iter().all(|&v| v.abs() < f64::EPSILON);
        if col1_all_zero && col2_all_zero {
            return false;
        }

        let all_same = mat
        .column(col1)
        .iter()
        .zip(mat.column(col2).iter())
        .all(|(&val1, &val2)| (val1 - val2).abs() < f64::EPSILON);

        if all_same {
            return false;
        }

        if let Some(prev) = prev_action {
            if prev == action {
                return false;
            }
        }
        true
    }

    pub fn get_valid_actions(
        &self,
        mat: &Array2<f64>,
        prev_action: Option<usize>,
    ) -> Vec<usize> {

        let candidates: Vec<usize> = (0..self.action_space).collect();
        let valid_actions: Vec<usize> = candidates
            .into_iter()
            .filter(|&action| self.is_valid_action(mat, action, prev_action))
            .collect();

        if valid_actions.is_empty() {
            (0..self.action_space).collect()
        } else {
            valid_actions
        }
    }

    fn swap_columns_and_rows<M>(mat: &mut ArrayBase<M, Ix2>, c1: usize, c2: usize)
    where
        M: DataMut<Elem = f64>,
    {
        for row in 0..mat.nrows() {
            mat.swap((row, c1), (row, c2));
        }
        for col in 0..mat.ncols() {
            mat.swap((c1, col), (c2, col));
        }
    }

    pub fn step(
        &mut self,
        mat: &Array2<f64>,
        action: usize,
        prev_action: Option<usize>,
    ) -> (Array2<f64>, bool, f64) {
        if self.is_done(mat) {
            return (mat.clone(), true, 0.0);
        }
        let valid_actions = self.get_valid_actions(mat, prev_action);
        let chosen_action = if valid_actions.contains(&action) {
            action
        } else {
            let mut rng = rand::thread_rng();
            *valid_actions.get(rng.gen_range(0..valid_actions.len())).unwrap()
        };
        let mut action_score = 0.0;
        let mut new_mat = mat.clone();

        let (mut col1, mut col2) = self.coupling_map[chosen_action];
        if col1 > col2 {
            std::mem::swap(&mut col1, &mut col2);
        }

        Self::swap_columns_and_rows(&mut new_mat, col1, col2);

        self.apply_coupling_sub(&mut new_mat);

        // gate ペナルティ
        action_score += self.gate;

        if self.used_columns_set.contains(&col1) || self.used_columns_set.contains(&col2) {
            self.reset_used_columns();
            action_score += self.layer_penalty;
        }

        self.used_columns_set.insert(col1);
        self.used_columns_set.insert(col2);

        let done = self.is_done(&new_mat);
        (new_mat, done, action_score)
    }

    fn apply_coupling_sub<M>(&self, mat: &mut ArrayBase<M, Ix2>)
    where
        M: DataMut<Elem = f64>,
    {
        for i in 0..mat.nrows() {
            for j in 0..mat.ncols() {
                let val = mat[[i, j]];
                let sub = val * self.coupling_map_mat[[i, j]];
                let new_val = val - sub;
                mat[[i, j]] = if new_val < 0.0 {
                    0.0
                } else if new_val > 1.0 {
                    1.0
                } else {
                    new_val
                };
            }
        }
    }

    pub fn get_reward(&self, mat: &Array2<f64>, total_score: f64) -> f64 {
        if self.is_done(mat) {
            1.0 - total_score
        } else {
            -1.0
        }
    }
}

#[pyclass]
pub struct PyGame {
    game: Game,
}

#[pymethods]
impl PyGame {
    #[new]
    pub fn new(qubits: usize, gate: f64, layer: f64, max_steps: usize) -> Self {
        let config = GameSettings {
            gate,
            layer,
            max_steps,
        };

        let game = Game::new(qubits, &config);

        PyGame { game }
    }

    #[pyo3(signature = (mat, prev_action=None))]
    pub fn get_valid_actions(
        &self,
        py: Python,
        mat: PyReadonlyArray2<f64>,
        prev_action: Option<usize>,
    ) -> PyResult<Vec<usize>> {
        let mat = mat.as_array().to_owned();
        let valid_actions = self.game.get_valid_actions(&mat, prev_action);
        Ok(valid_actions)
    }

    #[pyo3(signature = (mat, action, prev_action=None))]
    pub fn step(&mut self,py: Python, mat: PyReadonlyArray2<f64>, action: usize, prev_action: Option<usize>) -> PyResult<(Py<PyArray2<f64>>, bool, f64)> {
        let mat = mat.as_array().to_owned();
        let (new_mat, done, action_score) = self.game.step(&mat, action, prev_action);
        let py_new_mat = new_mat.into_pyarray_bound(py).to_owned();

        Ok((py_new_mat.into(), done, action_score))
    }

    #[pyo3(signature = (mat, total_score))]
    pub fn get_reward(
        &self,
        py: Python,
        mat: PyReadonlyArray2<f64>,
        total_score: f64,
    ) -> PyResult<f64> {
        let mat = mat.as_array().to_owned();
        let reward = self.game.get_reward(&mat, total_score);

        Ok(reward)
    }
}

#[pymodule]
fn game(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    Ok(())
}