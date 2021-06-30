### Reference:

https://blog.csdn.net/weixin_30699831/article/details/101982286

https://blog.csdn.net/halaoda/article/details/78661334

---------------------------------------------------------------
### 基础知识

git包括三个区域：工作区、暂存区、本地仓库，在远端（remote）包括：远程库

工作区(Work directory)：一般就是我们项目的根目录。

暂存区(Stage/Index)：版本库中设立一个暂存区(Stage/Index），作为用来直接跟工作区的文件进行交互，工作区文件的提交(commit)或者回滚(reset)都是通过暂存区，而版本库中除了暂存区之外，文件的提交的最终存储位置是分支(Branch)，在创建版本库的时候默认都会有一个主分支(Master)。

本地仓库(Repository)：我们在为项目添加本地库之后，会在工作区生成一个隐藏目录“.git”，.git目录即为当前工作区的本地版本库

版本控制流程：
1、修改本地已被跟踪文件(指旧文件)，文件进入未暂存区域
2、未暂存区域转到暂存区：`git add files`
3、暂存区提交到本地仓库：`git commit -m ‘commits’`
4、本地库回退到为暂存区：`git reset --mixed hash/origin/master`  tips: --mixed为默认参数，可以省略



### 基础命令

##### 1、添加文件到暂存区

添加指定文件：`git add file`  

添加工作路径下的所有修改文件：`git add .`

##### 2、将暂存区的修改提交到本地库

常规提交：`git commit -m 'commits'`

添加并提交(无法用于新文件)：`git commit -am ‘commits’`

取消add：`git reset HEAD` 回退到上一次操作

##### 3、回退版本(取消commit)的内容

回退到某一版本(commit)：`git reset [--soft | --mixed | --hard] [HEAD]`
**Notes:**
1、HEAD后跟^的数量表示回退的版本数，不加时用于取消暂存(add)的文件，直接返回当前的commit版本
2、--mixed为默认参数，可以省略，有此参数时会重置暂存区的文件与上一次的commit一致，工作区内容保持不变，并删除指定版本到当前所有的commit信息; 使用--hard参数时，会撤销工作区所有未提交的修改内容，并将暂存区和工作区都返回到上一版本，并删除指定版本到当前所有的commit信息。

##### 4、分支控制

查看分支：`git branch`

查看本地分支和远程分支：`git branch -a`

创建分支：`git branch mybranch`

切换分支：`git checkout mybranch` 

创建并切换分支：`git checkout -b mybranch`

##### 5、服务器端相关

显示已有服务器：`git remote`

显示服务器端的地址：`git remote -v`

添加服务器：`git add remote [name] ssh-address` 
tips:一般设置远端服务器名为origin

##### 6、其他

精简显示文件状态：`git status -s`  
tips:A表示新添加到暂存区的文件，M表示已修改，??表示未跟踪，靠左侧表示暂存区，靠右侧表示工作区

### 完全重建版本库

```bash
$ rm -rf .git 

$ git init 

$ git add . 

$ git commit -m "first commit"

$ git remote add origin <your_github_repo_url> 

$ git push -f -u origin master
```



### 开发分支（dev）合并到 master 分支

```bash
$ git checkout -b dev # 切换到开发分支

$ git pull # 将

$ git checkout master

$ git merge dev

$ git push -u origin master
```

