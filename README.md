# SimCLR Self supervised learning to learn ecgs embedding


### Usage

#### Train
- Make a copy of train_config.template.yaml and adjust your own parameters
- ```$ python main.py --app train --fname configs/train_config.yaml```

#### Finetune classifier
- Make a copy of finetune_config.template.yaml and adjust your own parameters
- ```$ python main.py --app finetune_classifier --fname configs/finetune_config.yaml```

#### Finetune nmt
- Make a copy of finetune_config.template.yaml and adjust your own parameters
- ```$ python main.py --app finetune_nmt --fname configs/finetune_config.yaml```



### Checkpoints

To do:
- Add more artificial transformations
- Add more ecgs like transformations
- Add support for running in distributed envirnoment