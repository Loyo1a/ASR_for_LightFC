from lib.test.evaluation.environment import EnvSettings


def local_env_settings(env_num):
    settings = EnvSettings()

    settings.davis_dir = r''
    settings.got10k_lmdb_path = r''
    settings.got10k_path = r''
    settings.got_packed_results_path = r''
    settings.got_reports_path = r''
    settings.itb_path = r''
    settings.lasot_extension_subset_path = ''
    settings.lasot_lmdb_path = r''
    settings.lasot_path = r'E:\LTSiam\dataset\LaSOT' #/media/liyunfeng/CV2/data/sot/lasot
    #settings.network_path = r'E:\LTSiam\LightFC-main\outputs\checkpoints\train\lightfc\tsl\LightFC_ep0400.pth.tar'
    settings.network_path = r'\outputs\checkpoints\train\lightfc\lightfc_asr_warmup_tinyvit\LightFC_ep0300.pth.tar'
    settings.nfs_path = r''
    settings.otb_path = r'D:\Drone\FedTrack-main\datasets\OTB100'
    settings.dtb_path = r'E:\LTSiam\dataset\DTB70'
    settings.prj_dir = r'E:\LTSiam\LightFC-main'
    settings.result_plot_path = r'E:\LTSiam\LightFC-main\outputs\test\result_plots'
    # Where to store tracking results
    settings.results_path = r'E:\LTSiam\LightFC-main\outputs\test\tracking_results'
    settings.save_dir = r'E:\LTSiam\LightFC-main\outputs'
    settings.segmentation_path = r'' #/home/liyunfeng/code/project2/LightFC/output/test/segmentation_results
    settings.tc128_path = r'' #/media/liyunfeng/CV2/data/sot/tc128
    settings.tn_packed_results_path = r''
    settings.tnl2k_path = r'' #/media/liyunfeng/CV2/data/sot/tnl2k/test
    settings.tpl_path = r''
    settings.trackingnet_path = r''
    settings.uav_path = r'E:\LTSiam\dataset\UAV123'
    settings.vot18_path = r''
    settings.vot22_path = r''
    settings.vot_path = r''
    settings.youtubevos_dir = r''
    settings.uot_path = r'' #/media/liyunfeng/CV2/data/uot/uot100
    settings.utb_path = r'' #/media/liyunfeng/CV2/data/uot/utb180

    settings.uavdt_path = r'E:\LTSiam\dataset\UAVDT'
    settings.visdrone2018_path = r'E:\LTSiam\dataset\VisDrone'
    settings.uavtrack_path = r'E:\LTSiam\dataset\UAVTrack'
    settings.uav20l_path = r'E:\LTSiam\dataset\UAV123'
    settings.uav123_10fps_path = r'E:\LTSiam\dataset\uav123_10fps'

    return settings
