import sys

from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000  # 限制 prompt的长度


def step_1_check_answer(state):
    # 判断第一个节点！有没有明确的answer回答 （item_name）
    # 1.获取 answer | is_stream
    answer = state.get("answer")
    is_stream = state.get("is_stream",False)
    if answer:
        # 有
        if is_stream:
            # 流式
            # 1. 推送到sse
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        else:
            # 非流式
            # 1. 设置任务结果
            set_task_result(state["session_id"], "answer", answer)
        # 流式
        # 非流式
        return True
    else:
        return False


def step_2_load_prompt(state):
    """
    加载模型润色答案的提示词
    :param state:
    :return: (prompt, sources_list) 提示词和引用来源列表
    """

    # 数据从state中获取
    rewritten_query = state.get("rewritten_query") or state.get("original_query")  # question
    reranked_docs = state.get("reranked_docs",[])
    item_names = state.get("item_names",[])
    history = state.get("history",[])

    # 1. 先处理 chunk块的内容 -》 context（带引用标识）
    docs = []
    sources = []  # 存储引用来源信息
    used_length = 0 #记录使用的长度

    for i, doc in enumerate(reranked_docs, start=1):
        text = doc.get("text")
        source = doc.get("source")  # local | web
        title = doc.get("title", "未知来源")
        url = doc.get("url", "")
        score = doc.get("score", 0.0)
        chunk_id = doc.get("chunk_id", "")

        # 构建context（带引用标识）
        content = f"[source:{i}] {text}"
        if used_length + len(content) > MAX_CONTEXT_CHARS:
            logger.info(f"本次内容停止追加了！已经大于限制长度！")
            break
        docs.append(content)

        # 记录引用来源信息
        source_info = {
            "index": i,
            "type": source,  # local | web
            "title": title,
            "url": url,
            "score": score,
            "chunk_id": chunk_id,
            "preview": text[:100] + "..." if len(text) > 100 else text
        }
        sources.append(source_info)
        used_length += len(content)

    final_context = "\n\n".join(docs)

    # 2. 再处理 history -> 聊天记录的内容
    history_str = "" # 对话记录的内容
    if history and len(history) > 0:
        for i,message in enumerate(history,start=1):
           role = message.get("role")
           text = message.get("text")
           current_history = ""
           if role == "user" and text:
               current_history = f"【用户】: {text}\n"
           elif role == "assistant" and text:
               current_history = f"【助手】: {text}\n"
           history_str += current_history
           used_length += len(current_history) #使用长度
           if used_length > MAX_CONTEXT_CHARS:
               logger.info(f"本次内容停止追加了！已经大于限制长度！")
               break
    else:
        history_str = "没有历史对话记录！"
    # 3. 再处理 item_name
    item_names_str =",".join(item_names)
    # 4. 再处理 question 问题
    answer_out_prompt = load_prompt("answer_out",
                           context=final_context,
                           history=history_str,
                           item_names=item_names_str,
                           question=rewritten_query)
    logger.info(f"已经完成了提示词生成：{answer_out_prompt}")
    return answer_out_prompt, sources


def step_3_create_answer(state, prompt):
    """
    使用模型生成最终的答案
    :param state:
    :param prompt:
    :return:
    """
    # 1. 获取模型对象和客户端
    model = get_llm_client()
    # 2. 获取流式状态【sse | set_result】
    is_stream = state.get("is_stream",False)
    answer = ''
    if is_stream:
        # 3. 调用模型进行生成 sse . stream  ||  set_result . invoke
        # 1 2 3 4 5 6 7
        for chunk in model.stream(prompt):
            # 3.1 推到sse
            delta = chunk.content # 1 | 2 3 | 4 | 5 6 7 |
            answer += delta #累加答案
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": delta})
    else:
        # 4. 最终的答案赋值给state['answer'] = 答案
        response = model.invoke(prompt)
        content = response.content
        answer = content
        set_task_result(state["session_id"], "answer", content)
    # 5. 返回结果answer即可
    state['answer'] = answer
    logger.info(f"lm模型最终返回的结果：{answer}")
    return answer


# 上标数字映射（1-9）
_SUPERSCRIPT_NUMBERS = ['¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']
# 圆圈数字映射（1-9）
CIRCLED_NUMBERS = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨']


def _clean_title(title: str) -> str:
    """清理标题中的Markdown标记"""
    # 去掉开头的 # 和空格
    cleaned = re.sub(r'^#{1,6}\s*', '', title.strip())
    return cleaned


def _format_citation_marker(indices: list) -> str:
    """
    将引用索引列表格式化为圆圈数字标记
    例如: [1, 2] -> "①②"
    """
    markers = []
    for idx in sorted(indices):
        if 1 <= idx <= 9:
            markers.append(CIRCLED_NUMBERS[idx - 1])
        else:
            markers.append(f'[{idx}]')
    return ''.join(markers)


def step_3_5_extract_citations(answer: str, sources: list) -> tuple:
    """
    从答案中提取引用的来源，构建引用列表，并将[source:N]替换为上标标记

    Args:
        answer: 包含[source:N]标记的答案文本
        sources: 原始来源列表（step_2_load_prompt生成）

    Returns:
        (citations, clean_answer) 引用列表和带引用标记的答案
    """
    # 支持多种引用格式：
    # [source:1] - 单个来源
    # [source:1,2] - 逗号分隔
    # [source:1, source:2] - 逗号+空格+source:分隔
    citation_pattern = re.compile(r'\[source:\s*(\d+(?:\s*,\s*(?:source:\s*)?\d+)*)\]')

    # 收集所有被引用的来源索引
    cited_indices = set()
    matches = citation_pattern.findall(answer)
    for match in matches:
        numbers = re.findall(r'\d+', match)
        for num in numbers:
            try:
                cited_indices.add(int(num))
            except ValueError:
                continue

    # 构建引用列表（清理标题中的#）
    citations = []
    for source in sources:
        if source["index"] in cited_indices:
            citations.append({
                "index": source["index"],
                "type": source["type"],
                "title": _clean_title(source["title"]),
                "url": source["url"],
                "score": source["score"],
                "chunk_id": source["chunk_id"],
                "preview": source["preview"]
            })

    # 替换答案中的[source:N]为圆圈数字标记
    def replace_citation(match):
        indices_str = match.group(1)
        indices = [int(x) for x in re.findall(r'\d+', indices_str)]
        return _format_citation_marker(indices)

    clean_answer = citation_pattern.sub(replace_citation, answer)
    # 清理多余空格
    clean_answer = re.sub(r'  +', ' ', clean_answer).strip()

    logger.info(f"提取到 {len(citations)} 个引用来源，索引: {cited_indices}")
    return citations, clean_answer


def step_4_extract_images_url(state):
    """
    从local -> chunk -> text中提取
       {text:" ![](url) "}  -> url
    从web   -> url   -> 图片提取
       mcp { url:"网络搜索 关联网址 || 图片地址" }
    :param state:
    :return:
    """
    images = []  # -> 存储图片 (On) (想要先后顺序)
    set_images = set() # -> 图片重复判断  （重复时间复杂度 O1）

    # 1. 定义正则
    image_reg = re.compile(r"!\[.*?\]\((.*?)\)")
    # reranked_docs => [{text,chunk_id,score,url,title,source local | web },{}]
    # 2. 宣传处理切片 -》 从高分 -》 低分
    reranked_docs = state.get("reranked_docs",[])
    for doc in reranked_docs:
       #{text,chunk_id,score,url,title,source local | web }
       # url -> 是不是图片
       url = doc.get("url")
       if url:
           if url.endswith((".png",".jpg",".jpeg",".gif",".webp")):
               # set -> not in  O1
               if url not in set_images:
                   images.append(url)
                   set_images.add(url)

       text = doc.get("text")
       # text -> 正则提取图片
       if text:
           # 正在匹配的所有图片
           matches = image_reg.findall(text)
           for image_url in matches:
               if image_url not in set_images:
                   images.append(image_url)
                   set_images.add(image_url)
       # 不存在-》添加到images即可
    logger.info(f"已经完成图片提取。数量:{len(images)},提取内容：{images}")
    state['image_urls'] =  images
    return images


def step_5_write_history(state):
    """
    将对话存储到mongodb history
    每次对话 对应2条history
       我们问  -》 user  ->  question -> text
       查询到  -》 assistant -> answer -> text
    :param state:
    :return:
    """
    session_id = state.get("session_id")
    answer = state.get("answer")
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    item_names = state.get("item_names",[])

    # if rewritten_query:
    #     # user
    #     save_chat_message(
    #         session_id = session_id,
    #         role = "user",
    #         text = rewritten_query,
    #         item_names = item_names
    #     )
    if answer:
        # assistant
        save_chat_message(
            session_id = session_id,
            role = "assistant",
            text = answer,
            item_names = item_names,
            rewritten_query=rewritten_query
        )
    logger.info(f"完成了本次对话的记录存储！")


def node_answer_output(state):
    """
    宏观：将最终topk -> 大模型 -> 润色 -> 结果 -> 【 【流式】 sse -》 前端 （push_to_session）  【非流式】set_task_result】
       1. 先检查state中是否存在answer回答  【item_name (1.明确 【 2.不确定 3.没有】 answer -> state)】 有可以直接写回答案
       2. 生成对应的润色的提示词 prompt（带引用来源）
       3. 使用模型润色答案 -》 结果 -> 文本（包含[source:N]标记）
       4. 提取引用来源列表，清理答案中的引用标记
       5. 提取原来topklist中的图片地址，单独返回【see】
       6. 对话的聊天记录（用户 user/助手 assistant）
       7. sse-final->返回图片和引用
    节点功能：进行过处理可以是流式输出可以整体输出！
    """
    print("---node_answer_output 节点处理开始---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    citations = []  # 引用来源列表

    # 1. 检查state中是否存在answer回答  【item_name (1.明确 【 2.不确定 3.没有】 answer -> state)】
    answer_exists = step_1_check_answer(state)
    if not answer_exists:
        # 2. 没有，生成对应的润色的提示词 prompt（带引用来源）
        prompt, sources = step_2_load_prompt(state)
        # 3. 没有，使用模型润色答案 -》 结果 -> 文本（包含[source:N]标记）
        raw_answer = step_3_create_answer(state, prompt)
        # 4. 提取引用来源列表，清理答案中的引用标记
        citations, clean_answer = step_3_5_extract_citations(raw_answer, sources)
        state['citations'] = citations
        # 更新state中的answer为清理后的版本（不含[source:N]标记）
        state['answer'] = clean_answer
        # 5. 提取原来topklist中的图片地址，单独返回【see】
        images_url = step_4_extract_images_url(state)
        # 6. sse-final->返回图片和引用
        final_data = {
            "answer": clean_answer,
            "status": "completed",
        }
        if images_url:
            final_data["image_urls"] = images_url
        if citations:
            final_data["citations"] = citations
        logger.info(f"SSE FINAL 推送数据: answer长度={len(clean_answer)}, citations数量={len(citations)}, images数量={len(images_url) if images_url else 0}")
        logger.info(f"SSE FINAL citations内容: {citations}")
        if state.get("is_stream", False):
            push_to_session(state["session_id"], SSEEvent.FINAL, final_data)

    # 数据都已经推送完毕了
    # 7. 添加聊天记录（mongodb）
    step_5_write_history(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    print("---node_answer_output 节点处理结束---")
    return {
        "answer": state.get("answer", ""),
        "image_urls": state.get("image_urls", []),
        "citations": state.get("citations", []),
    }


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://www.baidu.com/img/bd_logo.png)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",  # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None  # 初始无答案
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)

        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
