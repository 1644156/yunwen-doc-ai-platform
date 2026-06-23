import os
import sys

from pathlib import Path
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    设计的state: local_file_path [ is_read_md_enabled is_read_pdf_enabled ] md_path pdf_path file_title
    未来要实现:
       1.进入节点的日志输出 【节点 + 核心参数】
         记录任务状态 【哪个任务开始了】 -》 给前端推送信息 （埋点）
       2. 参数校验 （local_file_path -> 没有传入文件 -> end  / local_dir -> 没有传入输出文件夹 -> 创建一个临时）
       3. 解析文件类型，修改state对应的参数 local_file_path -> md | pdf
          -> is_md_read_enabled True  ||   is_pdf_read_enabled True
          -> md_path = local_file_path | pdf_path = local_file_path
          -> file_tile = 读取文件名
       4.结束节点的日志输出 【节点 + 核心参数】
         记录任务状态 【哪个任务结束了】 -》 给前端推送信息 （埋点）
    """
    # 1. 进入节点的日志输出 【节点 + 核心参数】 记录任务状态（给前端推送信息）
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state['task_id'],function_name)

    # 2. 进行必要的非空校验判定
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.error(f"[{function_name}]检查发现没有输入文件，无法继续解析！！")
        return state

    # 3. 判定并且完成state属性赋值
    if local_file_path.endswith(".md"):
        # 处理md
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif local_file_path.endswith(".pdf"):
        # 处理pdf
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.error(f"[{function_name}]文件格式不是md,pdf，无法继续解析！！")

    # 提取file_title  /xx/xxx/aaaa.pdf ->  aaa -> 为了后期大模型没有识别出来当前文件对应item_name -> file_title进行兜底
    #                               aaaa.pdf
    # os.path
    # aa.bb.tar.gz
    file_title_os = os.path.basename(local_file_path).split(".")[0]
    file_title = Path(local_file_path).stem # 去掉后缀的文件名  .name  .suffix
    state['file_title'] = file_title
    # 4. 结束节点的日志输出 【节点 + 核心参数】 记录任务状态（给前端推送信息）
    logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state}")
    add_done_task(state['task_id'], function_name)
    return state