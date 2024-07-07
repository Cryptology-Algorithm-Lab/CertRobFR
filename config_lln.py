from easydict import EasyDict as edict
config = edict()

config.output = "/* Your Workspace */"
config.lr = 0.001
config.weight_decay = 0
config.verbose = 2000
config.frequent = 20
config.num_workers = 4
config.batch_size = 512
config.rec = "/* Your directory for the train dataset /*"
config.num_classes = 93431
config.num_image = 5179510
config.emb_size = 512
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.fp16 = True
config.lambd = 0
config.xi = 0