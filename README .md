# jotang-recruit-2026-git

焦糖工作室 2026 招新 Git 题目的练习仓库。

目前用来记录 Git 基础操作的学习过程；后续会在这里放机器学习相关实验的代码和笔记。

## 仓库现状

当前仓库内容很少，`main` 分支已与远程 `origin/main` 对齐。

- 项目说明：本文件
- 提交记录：先写了 Git 学习笔记，再补充了真实操作里踩过的坑

机器学习实验相关文件还没有加入。

## Git 基础操作

### 1. 本地初始化

在 VS Code 中点击「初始化仓库」，把当前普通文件夹转换成 Git 可管理的版本仓库。

然后在终端配置提交身份，把名字和邮箱换成自己 GitHub 账号对应的信息：

```bash
git config --global user.name "GitHub 的用户名"
git config --global user.email "GitHub 绑定的邮箱地址"
```

### 2. 本地提交

```bash
git add 文件名
git commit -m "提交说明"
```

- `git add`：把指定文件加入暂存区，准备提交
- `git commit`：把暂存区的内容生成一条版本记录

### 3. 远程同步

```bash
git remote add origin 仓库地址
git push -u origin main
```

- `git remote add origin`：把本地仓库和远程 GitHub 仓库绑定起来
- `git push -u origin main`：把本地的 commit 推送到远程保存

## 踩坑记录

只在 GitHub 网页上拖拽上传文件，看起来仓库里有内容，但本地 Git 仓库里没有对应的版本历史。之后在 VS Code 里改代码再推送，会和远程历史冲突，推不上去。

这次实际遇到的情况：

1. 直接在 GitHub 网页里新建了内容，但没有把本地 Git 和远程仓库绑定起来，推送一直报错。
2. 后来又把文件拖到网页端仓库。远程已经有 README，本地文件夹却是另建的，两边历史完全脱节，本地 Git 认不上远程。
3. 最后删掉 GitHub 上那个对不上的仓库，改从本地 Git 重新推上去，两边才对齐。

正确做法是：先在本地完成初始化和提交，再绑定远程并 `git push`，让本地成为版本历史的起点。
