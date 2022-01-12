# 2021-shandong-Smart-Grid-Classification

下载官方数据放到data/shandong目录下
下载nezha-cn-base等权重放到user_data/pretrain_model/目录下
修改run_classify.py中的args里的model_path路径，为预训练最后保存的路径


先预训练，后微调 （run_pretrain.py -> run_classify.py）
