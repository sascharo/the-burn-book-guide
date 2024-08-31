/// The code defines a convolutional neural network model using the Rust burn library. It defines a
/// convolutional neural network that takes images as input, applies a series of convolutions,
/// pooling, and linear transformations, and outputs class scores for each input image. The specific
/// architecture and hyperparameters (e.g., number of channels, kernel sizes, dropout rate) are
/// configurable through the ModelConfig struct.
//? These imports bring in the necessary building blocks from the burn library to define, configure,
//? and utilize different layers and components within a neural network model.
use burn::{
    //? Imports several items from the nn (neural network) module within the burn crate.
    nn::{
        //? Imports the Conv2d struct (representing a 2D convolutional layer) and the Conv2dConfig
        //? struct (used to configure a Conv2d layer).
        conv::{ Conv2d, Conv2dConfig },
        pool::{
            //? Imports the AdaptiveAvgPool2d struct (representing an adaptive average pooling
            //? layer) and its corresponding configuration struct, AdaptiveAvgPool2dConfig.
            AdaptiveAvgPool2d,
            AdaptiveAvgPool2dConfig,
        },
        //? Imports the Dropout struct (for applying dropout regularization) and the DropoutConfig
        //? struct (for configuring dropout).
        Dropout,
        DropoutConfig,
        //? Linear, LinearConfig: Imports the Linear struct (representing a fully connected linear
        //? layer) and the LinearConfig struct (for configuring a Linear layer).
        Linear,
        LinearConfig,
        //? Relu: Imports the Relu struct, which represents the ReLU (Rectified Linear Unit)
        //? activation function.
        Relu,
    },
    //? The prelude module usually contains a collection of commonly used items that are intended to
    //? be easily accessible. By importing everything from the prelude, you can use these items
    //? without needing to specify their full paths.
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 10)]
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, class_prob]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}
