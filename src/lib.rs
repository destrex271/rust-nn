use std::cmp::max;

struct NeuralNode{
    inputs: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
    output: f64
}

pub trait NeuralFuncs{
    // ReLU Activation function
    fn activation_function(output: f64) -> f64{
        if output > 0.0{
            return output;
        }
        0.0
    }

    // Net output at neuron for a set of inputs and weights
    fn calculate_net(weights: Vec<f64>, inputs: Vec<f64>, bais: f64) -> f64{
        let mut sum: f64 = 0.0;
        for i in 1..weights.len(){
            sum += weights[i] * inputs[i];
        }
        sum += bais;
        return sum;
    }

    fn backpropogate(){}
}
