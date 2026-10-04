import numpy as np
score_dict={
    "张三":93,
    "李四":82,
    "王五":70,
    "许六":62,
    "陈七":96,
    "吴八":88,
    "付九":79
}

score = list(score_dict.values())
average = sum(score) / len(score)
average = round(average,1)
zg = max(score)

print("平均分:", average, "最高分:", zg)

a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([[7, 8],
              [9, 10],
              [11, 12]])
result = a @ b

print("输入矩阵A:\n", a)
print("矩阵A形状:\n", a.shape)
print("输入矩阵B:\n", b)
print("矩阵B形状:\n", b.shape)
print("矩阵乘法结果:\n", result)
print("结果矩阵形状:", result.shape)

