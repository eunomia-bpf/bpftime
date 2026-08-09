# 重现步骤（远程附加，在arm32上不工作）

1. 编译patch  

    ```console
    $ cd bpftime/workloads/ebpf-patch-dev/poc5-vim
    $ make
    poc.bpf.o 
    ```

    当前目录生成了 btf object文件。由于vim会触发pahole bug无法生成btf，因此我们不采用btf，而是直接人工将ufunc信息写到了json。

2. 重现漏洞  

    利用当前目录的prebuild vim来打开恶意文本触发漏洞：  

    ```console
    # step-1: 开启modeline模式。 
    $ touch ~/.vimrc
    $ echo "set modeline" > ~/.vimrc

    # step-2: 用vim打开poc.txt包含恶意文本
    $ ./vim poc.txt

    Press ENTER or type command to continue
    ```

    会发现vim无任何异常，按回车之后进入编辑界面是空白的，poc.txt仿佛是空文本。但是其内容是：
    > :!touch a.txt||" vi:fen:fdm=expr:fde=assert_fails("source\!\ \%"):fdl=0:fdt="

    会在当前目录创建一个a.txt文件。

3. 使用attach模式修复漏洞  

    step-1: shell-1中启动vim

    ```sh
    rm a.txt
    ./vim

    ```

    step-2: shell-2中，获取vim进程id，修改配置  

    ```console
    $ cd bpftime/workloads/ebpf-patch-dev/poc5-vim
    $  ps -ef | grep vim
    root       19151   18973  0 08:56 pts/2    00:00:00 ./vim
    ```

    将vim pid (19151) 填进 poc5.json。

    step-3: shell-3中，远程attach到vim进程，执行patch

    ```console
    $ cd /bpftime
    $ build/tools/cli/bpftime-cli workloads/ebpf-patch-dev/poc5-vim/poc5.json
    Successfully injected. ID: 1

    # shell-1的vim界面出现：
    register ufunc function: check_secure
    ~                                  register ufunc function: openscript                                        
    ~                                                                   find and load program: openscript       
    ~                                                                                                    load insn cnt: 56                                                                                                  
    ~         find function openscript at 0xaaaad9af7b44                                                        
    ~                                                   attach replace 0xaaaad9af7b44                           
    ~                                                                                Successfully attached 
    ```

    step-4: 打开poc.txt，去shell-1的vim界面，输入`:o poc.txt`，尝试触发漏洞  

    ```
    # shell-1 vim编辑窗口内容
    :!touch a.txt||" vi:fen:fdm=expr:fde=assert_fails("source\!\ \%"):fdl=0:fdt="                               
    ~                                                   attach replace 0xaaaad9af7b44                           
    ~                                                                                Successfully attached 
    78Copenscript: poc.txt 0
        find func check_secure at 0
                Unknown type: 0
                check_secure return 0filter op code: 0 ret: 0
    ```

    这时编辑窗口已经显示了poc.txt的内容，检查根目录，a.txt没有被创建。漏洞被成功修复。
