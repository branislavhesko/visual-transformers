from config.data_mode import Mode


class DINOConfig:
    decay = 0.996
    num_epochs = 10
    lr = 1e-3
    gamma = 0.97
    center_momentum = 1
    weights_momentum = 1
    temperature = 1
    device = "cuda"
    visualization_frequency = {
        Mode.train: 100,
        Mode.eval: 100
    }
    validation_frequency = 1
    model_name = "DINO"
