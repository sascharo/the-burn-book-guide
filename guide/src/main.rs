use lib::{ inference::infer, model::ModelConfig, training::{ train, TrainingConfig } };
use burn::{ backend, data::dataset::{ Dataset, vision::MnistDataset }, optim::AdamConfig };
use clap::{ command, Arg, Command };

fn parse_args() -> clap::ArgMatches {
    command!()
        .arg(
            Arg::new("artifact_dir")
                .short('a')
                .long("artifact")
                .help("Artifact directory")
                .required(false)
                .default_value("./artifact")
        )
        .arg(
            Arg::new("device")
                .short('d')
                .long("device")
                .help("Device")
                .required(false)
                .default_value("DiscreteGpu(0)")
        )
        .subcommand(Command::new("model").aliases(["print", "print-model"]).about("Print model"))
        .subcommand(
            Command::new("train")
                .about("Training")
                .arg(
                    Arg::new("num_epochs")
                        .short('e')
                        .long("epochs")
                        .help("Number of epochs")
                        .required(false)
                        .default_value("10")
                )
                .arg(
                    Arg::new("batch_size")
                        .short('b')
                        .long("batch_size")
                        .help("Batch size")
                        .required(false)
                        .default_value("64")
                )
                .arg(
                    Arg::new("num_workers")
                        .short('w')
                        .long("workers")
                        .help("Number of workers")
                        .required(false)
                        .default_value("4")
                )
                .arg(
                    Arg::new("learning_rate")
                        .short('l')
                        .long("lr")
                        .help("Learning rate")
                        .required(false)
                        .default_value("1.0e-4")
                )
        )
        .subcommand(
            Command::new("inf")
                .about("Inference")
                .arg(
                    Arg::new("index")
                        .short('i')
                        .long("index")
                        .alias("idx")
                        .help("Index")
                        .required(false)
                        .default_value("42")
                )
        )
        .get_matches()
}

fn main() {
    let matches = parse_args();
    if matches.subcommand_name().is_none() {
        println!("No valid subcommand provided.");
    }

    let device = backend::wgpu::WgpuDevice::default();
    println!("Device: {:?}", device);

    type MyBackend = backend::Wgpu<f32, i32>;
    type MyAutodiffBackend = backend::Autodiff<MyBackend>;

    let artifact_dir = matches
        .get_one::<String>("artifact_dir")
        .expect("Artifact directory is required.")
        .as_str();
    println!("Artifact directory: {}", artifact_dir);

    let model_config = ModelConfig::new(512);

    if let Some(_model_matches) = matches.subcommand_matches("model") {
        let model = model_config.init::<MyBackend>(&device);
        println!("Model:/n{}", model);
    } else if let Some(train_matches) = matches.subcommand_matches("train") {
        let num_epochs = train_matches
            .get_one::<String>("num_epochs")
            .unwrap()
            .parse::<usize>()
            .expect("Number of epochs must be an unsigned integer.");
        println!("Number of epochs: {:?}", num_epochs);
        let batch_size = train_matches
            .get_one::<String>("batch_size")
            .unwrap()
            .parse::<usize>()
            .expect("Batch size must be an unsigned integer.");
        println!("Number of epochs: {:?}", batch_size);
        let num_workers = train_matches
            .get_one::<String>("num_workers")
            .unwrap()
            .parse::<usize>()
            .expect("Number of epochs must be an unsigned integer.");
        println!("Number of workers: {:?}", num_workers);

        train::<MyAutodiffBackend>(
            artifact_dir,
            TrainingConfig::new(model_config, AdamConfig::new(), num_epochs),
            device.clone()
        );
    } else if let Some(infer_matches) = matches.subcommand_matches("inf") {
        let index = infer_matches
            .get_one::<String>("index")
            .unwrap()
            .parse::<usize>()
            .expect("Index must be an unsigned integer.");
        println!("Index: {:?}", index);

        infer::<MyBackend>(artifact_dir, device, MnistDataset::test().get(index).unwrap());
    }
}
