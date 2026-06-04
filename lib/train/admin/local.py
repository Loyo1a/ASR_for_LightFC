class EnvironmentSettings:
    def __init__(self, env_num=0):
        self.workspace_dir = r'E:\LTSiam\LightFC-main'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = r'E:\LTSiam\LightFC-main\tensorboard'  # Directory for tensorboard files.
        self.pretrained_networks = r'E:\LTSiam\LightFC-main\pretrained_models'

        self.lasot_dir = ''
        self.got10k_dir = r'E:\LTSiam\SiamFC\data\GOT-10k'
        self.got10k_val_dir = ''
        self.lasot_lmdb_dir = r'E:\LTSiam\dataset\LaSOT'
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = ''
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''

        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
