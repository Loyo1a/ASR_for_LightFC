import matplotlib.pyplot as plt
import sys
import os
plt.rcParams['figure.figsize'] = [8, 8]
env_num = 0
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from lib.test.analysis.plot_results import print_results, plot_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'otb' #utb


parameter_name = r'lightfc_asr_tinyvit' #lightfc_asr_warmup_tinyvit #mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou
trackers.extend(
    trackerlist(name='lightfc', parameter_name=parameter_name, dataset_name=dataset_name,
                run_ids=None, env_num=env_num, display_name=parameter_name))


dataset = get_dataset(dataset_name, env_num=env_num)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),
              env_num=env_num)
plot_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

