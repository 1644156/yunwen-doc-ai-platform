import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


"""
  主要目标： 将md中图片进行单独处理，方便后去模型识别图片的含义！
  主要动作： 图片->文件服务器-> 图片网络地址    （上文100）图片（下文100）->视觉模型-> 图片总结  
           ---》 [图片的总结](网络图片地址) -> state ->  md_content == 新的内容（图片处理后的）|| md_path = 处理后的md的地址
  总结技术：
        minio
        视觉模型： 提示词 + 访问 
  总结步骤： 
     1. 校验并且获取本次操作的数据 
        参数： state  -> md_path md_content 
        响应： 1. 校验后的md_content  2.md路径对象  3. 获取图片的文件夹 images
     2. 识别md中使用过的图片，采取做下一步（进行图片总结）
        参数： 1. md_content 2. images图片的文件夹地址
        响应： [(图片名,图片地址,(上文,下文))]
     3. 进行图片内容的总结和处理 （视觉模型）
        参数： 第二次的响应 [(图片名,图片地址,(上文,下文))]   || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
        响应： {图片名:总结,......}
     4. 上传图片minio以及更新md的内容 
        参数：minio_client || {图片名:总结,......} || [(图片名,图片地址,(上文,下文))] (minio) || md_content 旧 || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
        响应：new_md_content
        state[md_content] = new_md_content
     5. 进行数据的最终处理和备份 
        参数：new_md_content , 原md地址 -》 xx.md -> xx_new.md  
        响应：新的md的地址 new_md_path 
        state[md_path] = new_new_md_path
    return state
"""


def step_1_get_content(state:ImportGraphState) -> Tuple[str, Path, Path]:
    """
    提取内容
    :param state:
    :return:
    """
    # 1. 获取md的地址 md_path
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError("md_path不能为空！")

    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path:{md_file_path} 文件不存在！")

    # 要读取md_content
    if not state['md_content']:
        # 没有，再读取！ 有，证明是pdf节点解析过来的，已经给md_content进行赋值了！
        with md_path_obj.open("r", encoding="utf-8") as f:
            state['md_content'] = f.read()
    # 图片文件夹obj
    # 注意：自己传入的md -》 你的图片文件夹也必须交 images
    images_dir_obj = md_path_obj.parent / "images"
    return state['md_content'] , md_path_obj, images_dir_obj


def find_image_in_md_content(md_content, image_file,context_length:int=100):
    """
    从md_content识别图片的上下文！
    约定上下文长度100
    :param md_content:
    :param image_file:
    :param context_length：默认截取长度
    :return:
    """

    """
    # 你好啊
    我很好，还有7行代码今天就结束了！小伙伴们坚持好！谢谢！
    哈哈
    哈
    嘿嘿
    【start】 ![二大爷](/xxx/xx/zhaoweifeng.jpgxxx)【end】
    啦啦啦啦
    巴巴爸爸
    ![二大爷](/xxx/xx/zhaoweifeng.jpgxxx)
    嘿嘿额
    
    file_name zhaoweifeng.jpg
    """
    # 定义正则表达式  .*  .*?
    pattern = re.compile(r"!\[.*?\]\(.*?"+image_file+".*?\)")

    content = None #存储图片多处使用，上下文不同 ！ 本次暴力处理，获取第一个！
    # 查询符合位置
    items = list(pattern.finditer(md_content))
    if not items:
        return None
    if item := items[0]:
        start,end = item.span() #  span获取匹配对象的起始和终止的位置
        # 截取上文
        pre_text = md_content[max(start-context_length,0):start] # 考虑前面有没有context_length 没有从0开始
        post_text = md_content[end:min(end+context_length,len(md_content))] # 考虑后面有没有context_length 没有就到长度
        # 截取下文
        content = (pre_text,post_text)
    # 截取位置前后的内容
    if content:
        logger.info(f"图片：{image_file} ,在{md_content[:100]}，截取第一个上下文：{content}")
        return content

def step_2_scan_images(md_content:str, images_dir_obj:Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    进行md中图片识别，并且截取图片对应的上下文环境
    :param md_content:
    :param images_dir_obj:
    :return:  [(图片名，图片地址，上下元组())]
    """
    # 1. 我们先创建一个目标集合
    targets = []
    # 2. 循环读取images中的所有图片，校验在md中是否使用，使用了就截取上下文
    for image_file in os.listdir(images_dir_obj):
        # 遍历每个文件的名字
        # 检查图片是否可用 -》 图片
        if  not is_supported_image(image_file):
            logger.warning(f"当前文件：{image_file},不是图片格式，无需处理！")
            continue
        # 是图片，我们就在md查询，看是否存在，存在，读取对应的上下文即可
        # （上，下文）
        content_data = find_image_in_md_content(md_content,image_file)
        if not content_data:
            logger.warning(f"图片：{image_file}没有在md内容使用！上下文为空！")
            continue
        targets.append((image_file,str(images_dir_obj / image_file),content_data))

    return targets


def step_3_generate_img_summaries(targets, stem):
    """
    获取图片的内容描述！ 利用视觉模型！
    :param targets: [(图片名.xxx,图片地址,(上文,下文))，(图片名.xxx,图片地址,(上文,下文))]
    :param stem:  文件夹的名字  md名称 output / h180xxxx /  h180xxxx.md  | images
    :return: {图片名.xx : 总结和描述 , 图片名.xx : 总结和描述 , 图片名.xx : 总结和描述 ,图片名.xx : 总结和描述....}
    """
    summaries = {} # 最终结果
    # 循环每一张图片，向视觉模型进行请求，获取总结结果！
    # 确保一个对类对象就行了！！！
    request_times = deque()
    for image_file,image_path, context in  targets:
        # 解构 图片名 图片地址 (上,下)
        # 1. 访问限速问题（我们模型的限速标准 1分钟 可以访问10  限制并发访问次数..）
        apply_api_rate_limit(request_times, max_requests=500)
        # 2. 向视觉模型发起请求
        # 2.1 模型对象
        vm_model = get_llm_client(model=lm_config.lv_model)
        # 2.2 准备提示词
        prompt = load_prompt("image_summary",root_folder=stem,image_content=context)

        # import base64
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")  # 字节转成字符
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # 直接放图片的网络地址 "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                            # base64图片转后的字符串  jpg -> image/jpeg
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": f"{prompt}"},
                ],
            },
        ]
        # 2.3 执行获取总结
        response = vm_model.invoke(messages)
        summary = response.content.strip().replace("\n","")
        summaries[image_file] = summary
        logger.info(f"图片：{image_file}，总结结果：{summary}")
    logger.info(f"总结图片，获取结果：{summaries}")
    return summaries


def step_4_upload_images_and_replace_md(summaries, targets, md_content, stem):
    """
      将我们图片传递到minio服务器
      替换原md中的图片和描述
    :param summaries:  图片名 ： 描述
    :param targets:  （图片名，原地址，（上，下））
    :param md_content: 原md内容
    :param stem: 文件名
    :return: 新md
    """
    # 理解minio存储结果： 桶 / upload-images / 文件夹名字 / 图片对象.jpg
    minio_client = get_minio_client()
    # 1.  删除minio中的对应文件的图片
    # 1.1 获取要删除的对象
    # Object object_name
    # 注意：{minio_config.minio_img_dir[1:]}  一定要去掉一个 /
    object_list = minio_client.list_objects(minio_config.bucket_name,
                              prefix= f"{minio_config.minio_img_dir[1:]}/{stem}",
                              recursive=True)
    # 都有一个对象的名
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    # 需要的DeleteObject
    # 1.2 调用方法进行删除即可
    errors = minio_client.remove_objects(minio_config.bucket_name,delete_object_list)
    for errors in errors:
        logger.error(f"删除对象失败：{errors}")

    logger.info(f"已经完成{stem}下的对象清空，本次删除了：{len(delete_object_list)}个对象！！！")

    # 2. 上传图片到minio服务器
    # 声明记录图片上传结果的字典
    images_url = {}
    # targets:  （图片名，原地址，（上，下））
    for image_file,image_path, _ in targets:
        try:
            minio_client.fput_object(
                bucket_name= minio_config.bucket_name,
                object_name= f"{minio_config.minio_img_dir}/{stem}/{image_file}", # 传入minio 桶后面的命名  xx.png  xx/xxx/xx.png
                file_path= image_path,
                content_type="image/jpeg"
            )
            # 上传完毕以后记录
            # 图片地址 = 协议 + 端点 + 桶名 + 对象名  http://47.94.86.115:9000/ 桶名 / 对象名
            images_url[image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.info(f"完成图片{image_file}上传，访问地址为：{images_url[image_file]}")
        except Exception as e:
            logger.error(f"上传图片失败：{image_file}，失败原因：{e}")
    # 3. md中图片的替换即可
    # summaries = 图片名: 描述
    # images_url= 图片名：url地址
    # 汇总： {图片名:(描述,url地址)}
    image_infos = {}
    for image_file, summary in summaries.items():
        if url := images_url.get(image_file):
            image_infos[image_file] = (summary,url)
    logger.info(f"图片处理的汇总结果:{image_infos}")

    if image_infos:
        """
        xxxx
        xxx  ![xx](图片地址/image_file) -> ![summary](minio的url)
        xxx
        """
        for image_file, (summary, url) in image_infos.items():
            # 使用正则
            # ![](/xxx/xx/image_file) -> ![无所谓](无所谓image_file无所谓)
            rep = re.compile(r"!\[.*?\]\(.*?"+image_file+".*?\)")
            md_content = rep.sub(f"![{summary}]({url})", md_content)
        logger.info(f"已经完成md内容的替换，新的内容为:{md_content}")
    return md_content


def step_5_replace_md_and_save(new_md_content, md_path_obj):
    """
    完成新的md的磁盘本分，并且返回老地址！
    新的命名  xxx_new.md
    :param new_md_content: 新内容
    :param md_path_obj: 老地址
    :return: 新地址
    """
    # 设置下新的地址
    #   c:/xxx/xxx/xxx/xxxx/erdaye.md -> splitext(md_path_obj)[0]
    #   -》 c:/xxx/xxx/xxx/xxxx/erdaye _new.md
    new_md_path_str = os.path.splitext(md_path_obj)[0] + "_new.md"

    with open(new_md_path_str, "w", encoding="utf-8") as f:
        f.write(new_md_content)
    logger.info(f"已经完成了新内容的写入，新的地址为:{new_md_path_str}")
    return new_md_path_str


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state['task_id'], function_name)
    # 1. 校验并且获取本次操作的数据
    #         参数： state  -> md_path md_content
    #         响应： 1. 校验后的md_content  2.md路径对象  3. 获取图片的文件夹 images
    md_content, md_path_obj , images_dir_obj = step_1_get_content(state)
    # 如果没有图片，则直接返回 state
    if not images_dir_obj.exists():
        logger.info(f">>> [{function_name}]没有图片，直接返回 state ！")
        return state
    # 2. 识别md中使用过的图片，采取做下一步（进行图片总结）
    # [(图片名,图片地址,(上文,下文 = 100))]
    targets = step_2_scan_images(md_content, images_dir_obj)
    #         参数： 1. md_content 2. images图片的文件夹地址
    #         响应： [(图片名,图片地址,(上文,下文))]

    # 3. 进行图片内容的总结和处理 （视觉模型）
    # 参数： 第二次的响应 [(图片名,图片地址,(上文,下文))]   || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
    # 响应： {图片名:总结,......}
    summaries = step_3_generate_img_summaries(targets, md_path_obj.stem)
    # 4. 上传图片到minio同时替换md中的图片 （描述 + url地址）
    #         参数：minio_client || {图片名:总结,......} || [(图片名,图片地址,(上文,下文))] (minio) || md_content 旧 || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
    #         响应：new_md_content
    #         state[md_content] = new_md_content
    new_md_content = step_4_upload_images_and_replace_md(summaries, targets, md_content, md_path_obj.stem)

    # 5. 新的md内容替换和保存修改装
    #  参数：new_md_content , 原md地址 -》 xx.md -> xx_new.md
    #  响应：新的md的地址 new_md_path
    #  state[md_path] = new_new_md_path
    new_md_file_path = step_5_replace_md_and_save(new_md_content, md_path_obj)
    #  md_path -> 新的地址
    #  md_content -> 新的内容
    state["md_path"] = new_md_file_path
    state["md_content"] = new_md_content
    logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state}")
    add_done_task(state['task_id'], function_name)
    return state


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
