# Jotang-ml-task1 简单神经网络实验
这是焦糖工作室2026招新机器学习Task1的完整可运行实现，基于PyTorch从零搭建2层MLP，完成make_moons月牙数据集二分类全流程实验，覆盖所有任务要求的知识点。

## 环境依赖
运行前一键安装所有需要的库：
```bash
pip install torch numpy matplotlib scikit-learn
```

项目文件结构
Jotang-ml-task1/
    ├── mlp_moons.py # 完整MLP训练主代码
    ├── 任务1.md # 全部思考题解答+混淆矩阵与错误样本分析
    ├── README.md # 本说明文档
    └── output/ # 所有实验输出结果统一存放目录
        ├── 01_data_split.png # 训练集/测试集月牙数据分布拆分可视化
        ├── 02_baseline_curves.png # 基线正常训练的loss/准确率变化曲线
        ├── 03_decision_boundary.png # 基线模型的正常顺滑决策边界
        ├── 05_confusion_matrix.png # 正常测试集混淆矩阵
        ├── 06_errors_on_boundary.png # 边界错误样本标注可视化
        ├── 07_confusion_imbalanced.png # 不均衡数据集下的混淆矩阵
        ├── 08_boundary_imbalanced.png # 不均衡数据集下偏移的决策边界
        ├── 09_overfit_curves.png # 过拟合场景下loss/准确率变化曲线
        ├── 10_boundary_overfit.png # 过拟合模型的扭曲锯齿决策边界
        ├── ablation_activation.png # 三种激活函数的训练效果对照曲线
        ├── ablation_lr.png # 不同学习率的训练效果对照曲线
        ├── ablation_width.png # 不同隐藏层宽度的训练效果对照曲线
        ├── mlp_baseline.pt # 训练完成的基线模型权重文件
        └── summary.json # 所有对照实验的量化指标汇总表

## 运行方法
1.克隆仓库到本地
git clone https://github.com/CJY377/jotang-recruit-2026-git/
cd jotang-recruit-2026-git/Jotang-ml-task1
2.直接执行主脚本，自动跑完全部训练、绘图、指标计算流程：
python mlp_moons.py
3.运行结束后所有实验图表会自动保存到当前目录，终端会同步打印训练准确率、混淆矩阵、样本分布等全部关键实验结果。

## 已完成的核心实验 
1. 基础二分类训练：使用ReLU激活函数+CrossEntropyLoss损失函数，在make_moons数据集上达到98%以上的测试准确率，生成顺滑贴合月牙分布的正常决策边界
2. 三组完整变量对照实验：
    - 对照A：隐藏层宽度对照，依次测试h=4、h=32、h=128、h=512不同宽度下的训练收敛速度和最终准确率，直观验证模型容量对拟合效果的影响
    - 对照B：激活函数对照，在完全相同的训练条件下对比ReLU、Tanh、Sigmoid三种激活函数的收敛速度、最终精度差异
    - 对照C：学习率对照，固定使用SGD优化器，测试不同学习率下的loss曲线形态，复现学习率过大震荡不收敛、学习率过小下降极慢的典型现象
3. 过拟合复现实验：通过小训练集+复杂模型+关闭正则化+更长训练轮数，直观展示过拟合时验证损失反向上升、决策边界出现多余扭曲褶皱的典型特征
4. 正则化效果验证：对比`weight_decay=0`无正则化和`weight_decay=1e-4`开启L2正则化的训练效果差异，验证正则化对过拟合的抑制作用
5. 类别不均衡实验：构造1:20的极端不均衡数据集，验证全局准确率评估的局限性，展示决策边界向多数类偏移的现象
6. 混淆矩阵输出：统计所有测试样本的预测结果，挑选典型错误样本完成分布分析

## 所有实验的详细推导和结论都完整记录在任务1.md中