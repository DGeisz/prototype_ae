use crate::neurology::ProtoAENetwork;
use mccm::{MnistNetwork, MNIST_AREA};
use mnist::{Mnist, MnistBuilder};
use rand::Rng;

mod neurology;

const NUM_NEURONS: usize = 100;
const LEARNING_CONST: f32 = 0.001;
const EPOCHS: usize = 1;

const MIN_INIT_WEIGHT: f32 = 0.0;
const MAX_INIT_WEIGHT: f32 = 1.0 / NUM_NEURONS as f32;

const TRAINING_SET_LENGTH: u32 = 10000;
const TEST_SET_LENGTH: u32 = 1000;

fn generate_weight() -> f32 {
    rand::thread_rng().gen_range(MIN_INIT_WEIGHT, MAX_INIT_WEIGHT)
}

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(TRAINING_SET_LENGTH)
        .validation_set_length(0)
        .test_set_length(TEST_SET_LENGTH)
        .finalize();

    let train_img: Vec<f32> = trn_img.iter().map(|val| *val as f32 / 255.0).collect();
    let test_img: Vec<f32> = tst_img.iter().map(|val| *val as f32 / 255.0).collect();

    let mut network = ProtoAENetwork::new(NUM_NEURONS, MNIST_AREA, LEARNING_CONST, generate_weight);

    let accuracy = network.take_metric(train_img, trn_lbl, EPOCHS, test_img, tst_lbl);

    println!("Model accuracy: {}", accuracy);

    network.serialize();
}
