from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394', '396']

    if cfg.category == 'human_nerf' and cfg.task == 'zju_mocap':
        for sub in subjects:
            '''!!! use the comments below to revert to the default code if needed !!!'''
            dataset_attrs.update({
                f"zju_{sub}_train": {
                    "dataset_path": f"dataset/zju_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"zju_{sub}_test": {
                    "dataset_path": f"dataset/zju_mocap/{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap'
                },
                f"zju_{sub}_eval": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_eval",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap'
                },
            })

    subjects = {'zhanghongwei', 'zhuxuezhi', 'fuzhizhi2', 'ivan1', 'zhuna2', 'jinyutong2', 'dilshod', 'huajiangtao3', 'zhanghao'}
    if cfg.category == 'human_nerf' and cfg.task == 'genebody':
        for sub in subjects:
            '''!!! use the comments below to revert to the default code if needed !!!'''
            dataset_attrs.update({
                f"gb_{sub}_train": {
                    "dataset_path": f"dataset/genebody/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"gb_{sub}_test": {
                    "dataset_path": f"dataset/genebody/{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'genebody'
                },
                f"gb_{sub}_eval": {
                    "dataset_path": f"dataset/genebody/{sub}_eval",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'genebody'
                },
                f"gb_{sub}_eval_pose": {
                    "dataset_path": f"dataset/genebody/{sub}_eval_pose",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'genebody'
                },
                f"gb_{sub}_vis": {
                    "dataset_path": f"dataset/genebody/{sub}_vis",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'genebody'
                },
            })

    subjects = {'p000447_a000221', 'p000452_a000208', 'p000455_a000986', 'p000457_a000074', 'p000480_a000986', 'p000484_a000285', 'p000498_a000221', 'p000471_a000213', 'p000475_a000063', 'p100047_a001425'}
    if cfg.category == 'human_nerf' and cfg.task == 'humman':
        for sub in subjects:
            '''!!! use the comments below to revert to the default code if needed !!!'''
            dataset_attrs.update({
                f"hm_{sub}_train": {
                    "dataset_path": f"dataset/humman_selected/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                f"hm_{sub}_test": {
                    "dataset_path": f"dataset/humman_selected/{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'humman'
                },
                f"hm_{sub}_eval": {
                    "dataset_path": f"dataset/humman_selected/{sub}_eval",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'humman'
                },
                f"hm_{sub}_eval_pose": {
                    "dataset_path": f"dataset/humman_selected/{sub}_eval_pose",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'humman'
                },
                f"hm_{sub}_vis": {
                    "dataset_path": f"dataset/humman_selected/{sub}_vis",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'humman'
                },
            })

    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
