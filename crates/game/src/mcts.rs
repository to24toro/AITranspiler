use std::collections::HashMap;
use rand::Rng;
use ndarray::{Array2, Array1, Array, s};
use std::f64;
use rand::prelude::*;
use rand_distr::{Gamma, Distribution};
use crate::game::Game;

pub struct MctsSettings {
    pub dirichlet_alpha: f64,
    pub c_puct: f64,
    pub epsilon: f64,
    pub max_depth: usize,
    pub num_mcts_simulations: usize,
    pub tau_threshold: usize,
}

pub trait Network {
    fn predict(&self, state: &Array2<f64>) -> (Array1<f64>, f64);
}

pub struct MCTS<'a> {
    pub qubits: usize,
    pub network: &'a dyn Network,
    pub alpha: f64,
    pub c_puct: f64,
    pub epsilon: f64,
    pub max_depth: usize,
    pub num_mcts_simulations: usize,
    pub tau_threshold: usize,

    pub next_states: HashMap<String, Vec<Option<Array2<f64>>>>,
    pub P: HashMap<String, Array1<f64>>, // Policy
    pub N: HashMap<String, Array1<f64>>, // Visit counts
    pub W: HashMap<String, Array1<f64>>, // Total value
    pub V: HashMap<String, f64>,

    pub game: Game,

    pub initial_tau: f64,
    pub final_tau: f64,
    pub tau_decay_steps: usize,
    pub current_episode: usize,
}

impl<'a> MCTS<'a> {
    pub fn new(qubits: usize, network: &'a dyn Network, config: &MctsSettings, game: Game) -> Self {
        MCTS {
            qubits,
            network,
            alpha: config.dirichlet_alpha,
            c_puct: config.c_puct,
            epsilon: config.epsilon,
            max_depth: config.max_depth,
            num_mcts_simulations: config.num_mcts_simulations,
            tau_threshold: config.tau_threshold,

            next_states: HashMap::new(),
            P: HashMap::new(),
            N: HashMap::new(),
            W: HashMap::new(),
            V: HashMap::new(),

            game,

            initial_tau: 1.0,
            final_tau: 0.1,
            tau_decay_steps: 100,
            current_episode: 0,
        }
    }

    fn update_temperature(&self) -> f64 {
        let decay_factor = (self.current_episode as f64 / self.tau_decay_steps as f64).min(1.0);
        self.initial_tau * (1.0 - decay_factor) + self.final_tau * decay_factor
    }

    fn state_to_str(&self, state: &Array2<f64>) -> String {
        state
            .iter()
            .map(|&x| x.round().abs() as i32) 
            .map(|val| val.to_string())
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn search(&mut self, root_state: &Array2<f64>, num_simulations: usize, prev_action: Option<usize>)
        -> Array1<f64>
    {
        let tau = self.update_temperature();
        let s = self.state_to_str(root_state);

        if !self.P.contains_key(&s) {
            let _ = self.expand(root_state, prev_action);
        }

        let valid_actions = self.game.get_valid_actions(root_state, prev_action);

        // Dirichlet noise
        if !valid_actions.is_empty() {
            let dirichlet_noise = self.dirichlet_sampling(valid_actions.len(), self.alpha);
            // P[s][a] を (1-epsilon)*P + epsilon*noise で混合
            let mut p_s = self.P.get_mut(&s).unwrap();
            for (i, &action) in valid_actions.iter().enumerate() {
                let old_val = p_s[action];
                let new_val = (1.0 - self.epsilon) * old_val + self.epsilon * dirichlet_noise[i];
                p_s[action] = new_val;
            }
            let sum_p: f64 = p_s.sum();
            if sum_p > 0.0 {
                *p_s = p_s.mapv(|v| v / sum_p);
            }
        }

        for _sim in 0..num_simulations {
            let scores = {
                let p_s = self.P.get(&s).unwrap();
                let n_s = self.N.get(&s).unwrap();
                let w_s = self.W.get(&s).unwrap();
                let sum_n: f64 = n_s.sum() + 1e-8;

                let u: Vec<f64> = (0..self.game.action_space).map(|a| {
                    self.c_puct * p_s[a] * (sum_n.sqrt()) / (1.0 + n_s[a])
                }).collect();

                let q: Vec<f64> = (0..self.game.action_space).map(|a| {
                    if n_s[a] == 0.0 {
                        0.0
                    } else {
                        w_s[a] / n_s[a]
                    }
                }).collect();

                // スコア = u + q
                let mut scores_vec = vec![std::f64::NEG_INFINITY; self.game.action_space];
                let valid_acts = self.game.get_valid_actions(root_state, prev_action);
                for a in valid_acts.iter() {
                    scores_vec[*a] = u[*a] + q[*a];
                }

                scores_vec
            };

            let max_val = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let indices: Vec<usize> = scores.iter()
                .enumerate()
                .filter_map(|(i, &val)| if val == max_val { Some(i) } else { None })
                .collect();
            let chosen_action = {
                let mut rng = rand::thread_rng();
                if !indices.is_empty() {
                    *indices.get(rng.gen_range(0..indices.len())).unwrap()
                } else {
                    0
                }
            };

            if self.next_states[&s][chosen_action].is_none() {
                let (ns, done, score) = self.game.step(root_state, chosen_action, prev_action);
                self.next_states.get_mut(&s).unwrap()[chosen_action] = Some(ns.clone());
            }

            let next_state = self.next_states[&s][chosen_action].as_ref().unwrap();
            let v = self.evaluate(
                next_state,
                chosen_action,
                0,            // depth
                self.max_depth,
                0.0,         // total_score
            );

            {
                let w_s = self.W.get_mut(&s).unwrap();
                let n_s = self.N.get_mut(&s).unwrap();
                w_s[chosen_action] += v;
                n_s[chosen_action] += 1.0;
            }
        }

        // visit count から policy 作成
        let visits = self.N.get(&s).unwrap().clone(); // Array1<f64>
        let mut mcts_policy = visits.map(|x| if *x < 0.0 { 0.0 } else { *x });
        // tau > 0 なら mcts_policy^ (1/tau)
        for val in mcts_policy.iter_mut() {
            *val = val.powf(1.0 / tau);
        }
        // 正規化
        let sum_p: f64 = mcts_policy.sum();
        if sum_p > 0.0 {
            mcts_policy = mcts_policy.mapv(|v| v / sum_p);
        }

        mcts_policy
    }

    fn expand(&mut self, state: &Array2<f64>, prev_action: Option<usize>) -> f64 {
        let s = self.state_to_str(state);

        let (nn_policy, nn_value) = self.network.predict(state);

        self.P.insert(s.clone(), nn_policy.clone());
        self.N.insert(s.clone(), Array1::<f64>::zeros(self.game.action_space));
        self.W.insert(s.clone(), Array1::<f64>::zeros(self.game.action_space));
        self.next_states.insert(s.clone(), vec![None; self.game.action_space]);

        nn_value
    }

    fn evaluate(
        &mut self,
        state: &Array2<f64>,
        prev_action: usize,
        depth: usize,
        max_depth: usize,
        total_score: f64,
    ) -> f64 {
        if depth >= max_depth {
            return f64::NEG_INFINITY;
        }
    
        if self.game.is_done(state) {
            let reward = self.game.get_reward(state, total_score);
            return reward;
        }
    
        let s = self.state_to_str(state);
    
        if !self.P.contains_key(&s) {
            let nn_value = self.expand(state, Some(prev_action));
            return nn_value;
        } else {
            let valid_actions = self.game.get_valid_actions(state, Some(prev_action));
    
            let p_s = self.P.get(&s).unwrap();
            let n_s = self.N.get(&s).unwrap();
            let w_s = self.W.get(&s).unwrap();
    
            let sum_n = n_s.sum() + 1e-8;
    
            let u: Vec<f64> = (0..self.game.action_space)
                .map(|a| self.c_puct * p_s[a] * sum_n.sqrt() / (1.0 + n_s[a]))
                .collect();
    
            let q: Vec<f64> = (0..self.game.action_space)
                .map(|a| if n_s[a] == 0.0 { 0.0 } else { w_s[a] / n_s[a] })
                .collect();
    
            let mut scores_vec = vec![f64::NEG_INFINITY; self.game.action_space];
            for &a in valid_actions.iter() {
                scores_vec[a] = u[a] + q[a];
            }
    
            let max_val = scores_vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let best_actions: Vec<usize> = scores_vec
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val == max_val { Some(idx) } else { None })
                .collect();
            let chosen_action = {
                let mut rng = rand::thread_rng();
                if !best_actions.is_empty() {
                    best_actions[rng.gen_range(0..best_actions.len())]
                } else {
                    0
                }
            };
    
            if self.next_states[&s][chosen_action].is_none() {
                let (ns, done, sc) = self.game.step(state, chosen_action, Some(prev_action));
                self.next_states.get_mut(&s).unwrap()[chosen_action] = Some(ns);
            }
    
            let next_state = self.next_states[&s][chosen_action]
                .as_ref()
                .unwrap()
                .clone(); 
    
            let v = self.evaluate(&next_state, chosen_action, depth + 1, max_depth, total_score);
    
            let w_s_mut = self.W.get_mut(&s).unwrap();
            let n_s_mut = self.N.get_mut(&s).unwrap();
            w_s_mut[chosen_action] += v;
            n_s_mut[chosen_action] += 1.0;
    
            v
        }
    }

    fn dirichlet_sampling(length: usize, alpha: f64) -> Vec<f64> {
        let mut rng = thread_rng();
        let gamma = Gamma::new(alpha, 1.0).unwrap();
    
        let mut vals = (0..length)
            .map(|_| gamma.sample(&mut rng))
            .collect::<Vec<f64>>();
    
        let sum_ = vals.iter().sum::<f64>();
        if sum_ > 0.0 {
            for v in &mut vals {
                *v /= sum_;
            }
        }
        vals
    }

}


impl <'a> FromPyObject<'a> for MCTSConfig {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> PyResult<Self> {
        let dict = obj.downcast::<PyDict>()?;
        Ok(MCTSConfig { alpha: dict.get_item("alpha").and_then(|x| x.extract()).unwrap_or(0.5), c_puct: dict.get_item("c_puct").and_then(|x| x.extract()).unwrap_or(1.), epsilon: dict.get_item("epsilon").and_then(|x| x.extract()).unwrap_or(0.5), max_depth: dict.get_item("max_depth").and_then(|x| x.extract()).unwrap_or(40), num_mcts_simulations: dict.get_item("num_mcts_simulations").and_then(|x| x.extract()).unwrap_or(100) })
    }
}
