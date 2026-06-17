import os
from lib.utils.load import load_yaml
from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings


def parameters(yaml_name: str, env_num: int):
    params = TrackerParams()
    params.env_num = env_num
    prj_dir = env_settings(env_num).prj_dir
    save_dir = env_settings(env_num).save_dir
    # update default config from yaml file
    script_name = getattr(params, "script_name", None) or "lightfc"
    cfg_candidates = [
        os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (script_name, yaml_name)),
        os.path.join(prj_dir, 'experiments/lightfc_asr/%s.yaml' % yaml_name),
        os.path.join(prj_dir, 'experiments/lightfc/%s.yaml' % yaml_name),
    ]
    yaml_file = next((p for p in cfg_candidates if os.path.exists(p)), cfg_candidates[-1])
    params.cfg = load_yaml(yaml_file)
    print("test config: ", params.cfg)
    params.tracker_param = yaml_name

    # template and search region
    params.template_factor = params.cfg.TEST.TEMPLATE_FACTOR
    params.template_size = params.cfg.TEST.TEMPLATE_SIZE
    params.search_factor = params.cfg.TEST.SEARCH_FACTOR
    params.init_search_factor = getattr(params.cfg.TEST, "INIT_SEARCH_FACTOR",
                                        getattr(params.cfg.TEST, "INITIAL_SEARCH_FACTOR", params.search_factor))
    params.search_size = params.cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    ckpt_script = "lightfc_asr" if getattr(params.cfg.TEST, "ASR_ENABLE", False) else "lightfc"
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/%s/%s/LightFC_ep%04d.pth.tar" %
                                     (ckpt_script, yaml_name, params.cfg.TEST.EPOCH))
    if not os.path.exists(params.checkpoint):
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/%s/%s/lightfc_ep%04d.pth.tar" %
                                         (ckpt_script, yaml_name, params.cfg.TEST.EPOCH))

    params.save_all_boxes = False

    return params
