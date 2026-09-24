# jotang-recruit-2026-git
这个仓库是焦糖工作室2026招新Git题目的练习仓库，用来记录Git基础操作学习过程、后续机器学习相关实验的代码和笔记。

## Git操作学习笔记
1.  本地初始化流程
    - VScode里面点击*初始化仓库*：把当前普通文件夹转换成Git可管理的版本仓库
    - 配置我的提交身份：在下方终端里依次敲这两行命令，把名字和邮箱换成我GitHub账号对应的信息：
    git config --global user.name "Github的用户名"
    git config --global user.email "GitHub绑定的邮箱地址"


2.  基础提交流程
    - `git add 文件名`：把指定文件加入暂存区，准备提交
    - `git commit -m "提交说明"`：把暂存区的内容生成一条永久版本记录

3.  远程同步流程
    - `git remote add origin 仓库地址`：绑定本地仓库和远程GitHub仓库的关联
    - `git push -u origin main`：把本地的commit记录推送到远程GitHub仓库保存

4. 其实如果单纯提交的话从桌面拖拽到Github也可以，但是在本地的Git仓库里根本没有任何版本历史，后续你在VS Code里改代码想同步，会直接和远程仓库的历史冲突，根本推不上去