use mccm::{MnistNetwork, MnistNeuron, MNIST_SIDE};
use std::cell::RefCell;
use std::rc::Rc;

struct Synapse {
    measure: RefCell<f32>,
    weight: RefCell<f32>,
}

impl Synapse {
    fn new(measure: f32, weight: f32) -> Synapse {
        Synapse {
            measure: RefCell::new(measure),
            weight: RefCell::new(weight),
        }
    }

    /// First return is the weighted measure, second
    /// return is the synapse weight
    fn get_weighted_measure(&self) -> (f32, f32) {
        (
            *self.measure.borrow() * *self.weight.borrow(),
            *self.weight.borrow(),
        )
    }
}

pub struct ProtoAENeuron {
    name: String,
    synapses: Vec<Synapse>,
}

impl ProtoAENeuron {
    pub fn new(name: String, num_synapses: usize, weight_generator: fn() -> f32) -> ProtoAENeuron {
        let synapses = (0..num_synapses)
            .map(|_| Synapse::new(0.0, weight_generator()))
            .collect::<Vec<Synapse>>();

        ProtoAENeuron { name, synapses }
    }
}

impl MnistNeuron for ProtoAENeuron {
    fn load_val(&self, x: usize, y: usize, val: f32) {
        let val_i = (y * MNIST_SIDE) + x;
        let syn = self.synapses.get(val_i).unwrap();

        *syn.measure.borrow_mut() = val;
    }

    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn compute_em(&self) -> f32 {
        let mut total_weighted_measure = 0.0;
        let mut total_weights = 0.0;

        for syn in self.synapses.iter() {
            let (weighted_measure, weight) = syn.get_weighted_measure();

            total_weighted_measure += weighted_measure;
            total_weights += weight;
        }

        total_weighted_measure / total_weights
    }
}

pub struct ProtoAENetwork {
    neurons: Vec<Rc<ProtoAENeuron>>,
    learning_constant: f32,
}

impl ProtoAENetwork {
    pub fn new(
        num_neurons: usize,
        num_synapses: usize,
        learning_constant: f32,
        weight_generator: fn() -> f32,
    ) -> ProtoAENetwork {
        let neurons = (0..num_neurons)
            .map(|i| {
                Rc::new(ProtoAENeuron::new(
                    i.to_string(),
                    num_synapses,
                    weight_generator,
                ))
            })
            .collect::<Vec<Rc<ProtoAENeuron>>>();

        ProtoAENetwork {
            learning_constant,
            neurons
        }
    }
}

impl MnistNetwork for ProtoAENetwork {
    fn get_neurons(&self) -> Vec<Rc<dyn MnistNeuron>> {
        self.neurons
            .iter()
            .map(|neuron| Rc::clone(neuron) as Rc<dyn MnistNeuron>)
            .collect()
    }

    fn perform_adjustment(&mut self) {
        let mut total_weight_vec = Vec::new();
        let mut total_weighted_measure_vec = Vec::new();

        for neuron in self.neurons.iter() {
            let mut total_weight = 0.0;
            let mut total_weighted_measure = 0.0;

            for syn in neuron.synapses.iter() {
                let (weighted_measure, weight) = syn.get_weighted_measure();

                total_weighted_measure += weighted_measure;
                total_weight += weight;
            }

            total_weighted_measure_vec.push(total_weighted_measure);
            total_weight_vec.push(total_weight);
        }

        let weighted_average_vec = total_weight_vec
            .iter()
            .zip(total_weighted_measure_vec.iter())
            .map(|(&weight, &weighted_measure)| weighted_measure / weight)
            .collect::<Vec<f32>>();

        let first_neuron = self.neurons.get(0).unwrap();

        let mut zeta_vec = Vec::new();
        let num_inputs = first_neuron.synapses.len();
        let num_neurons = self.neurons.len();

        for t in 0..num_inputs {
            let mut weighted_prediction = 0.0;

            for j in 0..num_neurons {
                weighted_prediction += weighted_average_vec.get(j).unwrap()
                    * *self
                        .neurons
                        .get(j)
                        .unwrap()
                        .synapses
                        .get(t)
                        .unwrap()
                        .weight
                        .borrow();
            }

            let zeta =
                weighted_prediction - *first_neuron.synapses.get(t).unwrap().measure.borrow();

            zeta_vec.push(zeta);
        }

        for (a, neuron) in self.neurons.iter().enumerate() {
            let &neuron_weighted_avg = weighted_average_vec.get(a).unwrap();

            for (b, syn) in neuron.synapses.iter().enumerate() {
                let mut neuron_shared_term = 0.0;

                for t in 0..num_inputs {
                    neuron_shared_term += 2.0
                        * zeta_vec.get(t).unwrap()
                        * *neuron.synapses.get(t).unwrap().weight.borrow()
                        * (*first_neuron.synapses.get(a).unwrap().measure.borrow()
                            - neuron_weighted_avg);
                }

                *syn.weight.borrow_mut() -= (neuron_shared_term
                    + (2.0 * zeta_vec.get(b).unwrap() * neuron_weighted_avg))
                    * self.learning_constant;
            }
        }
    }
}
