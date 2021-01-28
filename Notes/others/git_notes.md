# Reference:

https://blog.csdn.net/weixin_30699831/article/details/101982286

https://blog.csdn.net/halaoda/article/details/78661334

---------------------------------------------------------------
# git bash:
### Concept   
修改的文件位于工作区，`add`后文件被加载进暂存区，`commit`后文件进入版本库

---

### Basic
1.从远程主机的特定分支获取更新，并合并到本地分支上   
`git pull {远程主机名} {远程分支}:{本地分支}`   

2.将本地的commit过的更新同步到远程主机   
`git push {远程主机名} {本地分支}:{远程分支}`

3.克隆版本库   
`git clone {远程地址}` 克隆整个版本库   
`git clone --bare {远程地址}` 仅克隆一个裸版本库

---

### Branch
1.实现为远程和本地两个分支建立关联   
`git branch --set-upstream-to=origin/{远程分支} {本地分支}`

2.创建新的分支   
`git branch {新的分支}`

3.查看所有分支   
`git branch -a`

4.删除分支   
`git branch -d {需要删除的分支}`

5.切换分支   
`git checkout {需要切换的分支}`

6.创建分支并切换到该分支 git branch + git checkout
`git checkout -b {新分支名}`

---

### Version Control
1.查看版本改动历史   
`git log` 显示历史记录   
`git log --oneline` 显示简略版历史记录   
`git log --graph` 显示图形化版本历史记录   
`git log -n {num}` 显示最后num次提交的commits   
`git log --dirstat` 显示被修改文件的目录   
`git log --shortstat` 显示多少文件被修改   

2.显示修改文件   
`git status` 查看发生修改   
`git status --short` 显示简略版修改   

3.显示提交的差异   
`git diff` 显示工作区未注册的本地修改   
`git diff {hash} Head` 显示两次提交的差异   
`git diff {hash}^!` 与上次提交进行比较   

4、从暂存区撤回修改（将add取消）   
`git reset HEAD {file}` 撤回指定文件的add
`git reset HEAD .` 撤回当前的所有修改

### Note:   
1.git pull应该放在commit -m和git push之间。  
如果存在add的内容没有commit，则不允许git pull

---

# Github:
在github端不能直接删除仓库中的文件，只能删除整个仓库

github中存有HSS和HTML两种地址

每次更换新的电脑、每次更换新的仓库，如果需要使用git的pull和push功能时，都需要将公钥加入到github自有库中的setting-keys中

---

## git&&pycharm（暂时不用了）:

### 设置

1、建立连接首先需要在Settings-Version_Control-git中，选定git.exe的所在地址

2、接着在Settings-Version_Control-github中，输入用户名和密码，进行连接

3、在VCS-Get_From_Version_Control中输入github库给定的HSS地址，并输入所需要克隆的位置

### 使用

1、修改工程需要进行同步时，首先右键工程点击Git-Add(可以设置自动添加）

2、add后，点击右上角commit，并输入相关信息

3、commit后，点击右上角的push，即可将更新同步到远端服务器

4、另一台电脑需要将本机的更新获得同步时，只要点击右上角的update(pull)即可
