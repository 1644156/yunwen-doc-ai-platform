import http.client
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import webbrowser

# API配置
API_HOST = "8.130.92.93"
API_KEY = "test-free"


class APIClient:
    def __init__(self):
        self.host = API_HOST
        self.api_key = API_KEY

    def _request(self, method, path, body=None, timeout=120):
        conn = http.client.HTTPConnection(self.host, timeout=timeout)
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            body_bytes = json.dumps(body).encode("utf-8") if body else None
            conn.request(method, path, body=body_bytes, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            return json.loads(data.decode("utf-8"))
        finally:
            conn.close()

    def list_models(self):
        try:
            result = self._request("GET", "/v1/models", timeout=30)
            return result.get("data", [])
        except Exception:
            return []

    @staticmethod
    def categorize_models(models):
        categories = {"chat": [], "image": [], "video": []}
        for m in models:
            mid = m.get("id", "")
            name = m.get("display_name", mid)
            if "video" in mid:
                categories["video"].append({"id": mid, "name": name})
            elif "image" in mid or "imagine" in mid:
                categories["image"].append({"id": mid, "name": name})
            else:
                categories["chat"].append({"id": mid, "name": name})
        return categories

    def chat_completion(self, prompt, model="grok-4.20-auto",
                        temperature=0.7, max_tokens=2048):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        try:
            result = self._request("POST", "/v1/chat/completions", body, timeout=120)
            if "choices" in result and result["choices"]:
                msg = result["choices"][0].get("message", {})
                content = msg.get("content", "")
                reasoning = msg.get("reasoning_content", "")
                if reasoning:
                    return f"[思考过程]\n{reasoning}\n\n[回答]\n{content}"
                return content or "无响应内容"
            return "无响应内容"
        except Exception as e:
            return f"请求失败: {e}"

    def generate_image(self, prompt, model="grok-imagine-image-pro",
                       size="1024x1024"):
        body = {"model": model, "prompt": prompt, "n": 1, "size": size}
        try:
            return self._request("POST", "/v1/images/generations", body, timeout=180)
        except Exception as e:
            return {"error": f"请求失败: {e}"}

    def generate_video(self, prompt, model="grok-imagine-video",
                       image_path=None):
        if image_path:
            try:
                boundary = "----PyBoundary"
                parts = b""
                for k, v in {"model": model, "prompt": prompt}.items():
                    parts += f"--{boundary}\r\n".encode()
                    parts += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
                    parts += f"{v}\r\n".encode()
                with open(image_path, "rb") as f:
                    file_data = f.read()
                fname = image_path.split("/")[-1].split("\\")[-1]
                parts += f"--{boundary}\r\n".encode()
                parts += f'Content-Disposition: form-data; name="image"; filename="{fname}"\r\n'.encode()
                parts += b"Content-Type: application/octet-stream\r\n\r\n"
                parts += file_data + b"\r\n"
                parts += f"--{boundary}--\r\n".encode()

                conn = http.client.HTTPConnection(self.host, timeout=300)
                try:
                    conn.request("POST", "/v1/videos/generations", body=parts, headers={
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                        "Authorization": f"Bearer {self.api_key}"
                    })
                    resp = conn.getresponse()
                    return json.loads(resp.read().decode("utf-8"))
                finally:
                    conn.close()
            except Exception as e:
                return {"error": f"请求失败: {e}"}
        else:
            body = {"model": model, "prompt": prompt, "n": 1}
            try:
                return self._request("POST", "/v1/videos/generations", body, timeout=300)
            except Exception as e:
                return {"error": f"请求失败: {e}"}


class APIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Grok API 工具")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        self.client = APIClient()
        self._busy = False
        self._setup_style()
        self._create_widgets()
        # 延迟到mainloop启动后再加载模型
        self.root.after(200, self._load_models)

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook.Tab", padding=[12, 6])
        style.configure("TButton", padding=[8, 4])

    def _create_widgets(self):
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(fill="x")
        self.connection_var = tk.StringVar(value="未连接")
        ttk.Label(toolbar, text="API:").pack(side="left")
        ttk.Label(toolbar, textvariable=self.connection_var,
                  foreground="gray").pack(side="left", padx=(2, 10))
        ttk.Button(toolbar, text="刷新模型", command=self._load_models).pack(side="left")
        ttk.Button(toolbar, text="测试连接", command=self._test_connection).pack(side="left", padx=5)

        self.notebook = ttk.Notebook(self.root, padding=5)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        self.chat_tab = ttk.Frame(self.notebook, padding=10)
        self.image_tab = ttk.Frame(self.notebook, padding=10)
        self.video_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.chat_tab, text="  文字聊天  ")
        self.notebook.add(self.image_tab, text="  文生图  ")
        self.notebook.add(self.video_tab, text="  文生视频  ")

        self._create_chat_tab()
        self._create_image_tab()
        self._create_video_tab()

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief="sunken", anchor="w", padding=[5, 2]).pack(fill="x", side="bottom")

    def _create_chat_tab(self):
        tab = self.chat_tab
        mf = ttk.LabelFrame(tab, text="模型", padding=8)
        mf.pack(fill="x", pady=(0, 8))
        self.chat_model_var = tk.StringVar(value="grok-4.20-auto")
        self.chat_model_combo = ttk.Combobox(mf, textvariable=self.chat_model_var, state="readonly", width=40)
        self.chat_model_combo.pack(side="left", fill="x", expand=True)

        pf = ttk.LabelFrame(tab, text="参数", padding=8)
        pf.pack(fill="x", pady=(0, 8))
        r1 = ttk.Frame(pf); r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Temperature:").pack(side="left")
        self.chat_temp_var = tk.DoubleVar(value=0.7)
        s = ttk.Scale(r1, from_=0, to=2, variable=self.chat_temp_var, orient="horizontal")
        s.pack(side="left", fill="x", expand=True, padx=5)
        self.chat_temp_label = ttk.Label(r1, text="0.7", width=5)
        self.chat_temp_label.pack(side="left")
        s.configure(command=lambda v: self.chat_temp_label.configure(text=f"{float(v):.1f}"))
        r2 = ttk.Frame(pf); r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Max Tokens:").pack(side="left")
        self.chat_maxtokens_var = tk.IntVar(value=2048)
        ttk.Spinbox(r2, from_=100, to=32000, increment=100,
                     textvariable=self.chat_maxtokens_var, width=10).pack(side="left", padx=5)

        inf = ttk.LabelFrame(tab, text="输入", padding=8)
        inf.pack(fill="x", pady=(0, 8))
        self.chat_prompt = tk.Text(inf, height=5, wrap="word")
        self.chat_prompt.pack(fill="x", pady=(0, 5))
        bf = ttk.Frame(inf); bf.pack(fill="x")
        self.chat_btn = ttk.Button(bf, text="发送", command=self._submit_chat)
        self.chat_btn.pack(side="left")
        ttk.Button(bf, text="清空", command=lambda: self.chat_prompt.delete("1.0", "end")).pack(side="left", padx=5)

        rf = ttk.LabelFrame(tab, text="结果", padding=8)
        rf.pack(fill="both", expand=True)
        self.chat_result = scrolledtext.ScrolledText(rf, wrap="word", state="disabled")
        self.chat_result.pack(fill="both", expand=True, pady=(0, 5))
        ttk.Button(rf, text="复制结果", command=lambda: self._copy_widget(self.chat_result)).pack(anchor="e")

    def _create_image_tab(self):
        tab = self.image_tab
        mf = ttk.LabelFrame(tab, text="模型", padding=8)
        mf.pack(fill="x", pady=(0, 8))
        self.image_model_var = tk.StringVar(value="grok-imagine-image-pro")
        self.image_model_combo = ttk.Combobox(mf, textvariable=self.image_model_var, state="readonly", width=40)
        self.image_model_combo.pack(side="left", fill="x", expand=True)

        pf = ttk.LabelFrame(tab, text="参数", padding=8)
        pf.pack(fill="x", pady=(0, 8))
        ttk.Label(pf, text="尺寸:").pack(side="left")
        self.image_size_var = tk.StringVar(value="1024x1024")
        ttk.Combobox(pf, textvariable=self.image_size_var, state="readonly", width=15,
                     values=["512x512", "768x768", "1024x1024", "1024x1792", "1792x1024"]).pack(side="left", padx=5)

        inf = ttk.LabelFrame(tab, text="输入", padding=8)
        inf.pack(fill="x", pady=(0, 8))
        ttk.Label(inf, text="提示词:").pack(anchor="w")
        self.image_prompt = tk.Text(inf, height=4, wrap="word")
        self.image_prompt.pack(fill="x", pady=(0, 5))
        bf = ttk.Frame(inf); bf.pack(fill="x")
        self.image_btn = ttk.Button(bf, text="生成图片", command=self._submit_image)
        self.image_btn.pack(side="left")
        ttk.Button(bf, text="清空", command=lambda: self.image_prompt.delete("1.0", "end")).pack(side="left", padx=5)

        rf = ttk.LabelFrame(tab, text="结果", padding=8)
        rf.pack(fill="both", expand=True)
        self.image_result = scrolledtext.ScrolledText(rf, wrap="word", state="disabled", height=6)
        self.image_result.pack(fill="x", pady=(0, 5))
        lf = ttk.Frame(rf); lf.pack(fill="x")
        ttk.Label(lf, text="链接:").pack(side="left")
        self.image_link_var = tk.StringVar()
        ttk.Entry(lf, textvariable=self.image_link_var, width=60).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(lf, text="打开", command=lambda: self._open_link(self.image_link_var)).pack(side="left")
        ttk.Button(lf, text="复制", command=lambda: self._copy_text(self.image_link_var.get())).pack(side="left", padx=2)

    def _create_video_tab(self):
        tab = self.video_tab
        mf = ttk.LabelFrame(tab, text="模型", padding=8)
        mf.pack(fill="x", pady=(0, 8))
        self.video_model_var = tk.StringVar(value="grok-imagine-video")
        self.video_model_combo = ttk.Combobox(mf, textvariable=self.video_model_var, state="readonly", width=40)
        self.video_model_combo.pack(side="left", fill="x", expand=True)

        inf = ttk.LabelFrame(tab, text="输入", padding=8)
        inf.pack(fill="x", pady=(0, 8))
        ttk.Label(inf, text="提示词:").pack(anchor="w")
        self.video_prompt = tk.Text(inf, height=4, wrap="word")
        self.video_prompt.pack(fill="x", pady=(0, 5))
        ref = ttk.Frame(inf); ref.pack(fill="x", pady=(0, 5))
        self.video_image_path = tk.StringVar()
        ttk.Label(ref, text="参考图片(可选):").pack(side="left")
        ttk.Entry(ref, textvariable=self.video_image_path, width=40).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(ref, text="浏览", command=self._browse_video_image).pack(side="left")
        bf = ttk.Frame(inf); bf.pack(fill="x")
        self.video_btn = ttk.Button(bf, text="生成视频", command=self._submit_video)
        self.video_btn.pack(side="left")
        ttk.Button(bf, text="清空", command=lambda: self.video_prompt.delete("1.0", "end")).pack(side="left", padx=5)

        rf = ttk.LabelFrame(tab, text="结果", padding=8)
        rf.pack(fill="both", expand=True)
        self.video_result = scrolledtext.ScrolledText(rf, wrap="word", state="disabled", height=6)
        self.video_result.pack(fill="x", pady=(0, 5))
        lf = ttk.Frame(rf); lf.pack(fill="x")
        ttk.Label(lf, text="链接:").pack(side="left")
        self.video_link_var = tk.StringVar()
        ttk.Entry(lf, textvariable=self.video_link_var, width=60).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(lf, text="打开", command=lambda: self._open_link(self.video_link_var)).pack(side="left")
        ttk.Button(lf, text="复制", command=lambda: self._copy_text(self.video_link_var.get())).pack(side="left", padx=2)

    # ---- 核心：同步请求 + 手动泵送事件循环 ----

    def _sync_request(self, func, *args):
        """在主线程中同步执行func，同时持续泵送tkinter事件循环保持界面响应"""
        self._busy = True
        self._set_buttons_state("disabled")

        result_holder = [None]
        error_holder = [None]
        done = []

        def _worker():
            try:
                result_holder[0] = func(*args)
            except Exception as e:
                error_holder[0] = e
            done.append(True)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # 主线程持续泵送事件循环，直到后台线程完成
        while not done:
            self.root.update()       # 处理所有待处理的tkinter事件
            time.sleep(0.05)         # 休眠50ms，避免CPU空转

        self._busy = False
        self._set_buttons_state("normal")

        if error_holder[0]:
            raise error_holder[0]
        return result_holder[0]

    def _set_buttons_state(self, state):
        self.chat_btn.config(state=state)
        self.image_btn.config(state=state)
        self.video_btn.config(state=state)

    # ---- 辅助方法 ----

    def _set_result(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def _copy_widget(self, widget):
        text = widget.get("1.0", "end-1c")
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def _copy_text(self, text):
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def _open_link(self, var):
        url = var.get()
        if url:
            webbrowser.open(url)

    def _browse_video_image(self):
        path = filedialog.askopenfilename(
            title="选择参考图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.gif *.bmp"), ("所有文件", "*.*")]
        )
        if path:
            self.video_image_path.set(path)

    def _get_model_id(self, name, id_map):
        return id_map.get(name, name)

    def _extract_url(self, result, media_type):
        if isinstance(result, str):
            return "", result
        if "error" in result:
            return "", f"请求失败: {result['error']}"
        if "data" in result and result["data"]:
            item = result["data"][0]
            url = item.get("url", "") or item.get("b64_json", "")
            if url and not url.startswith("data:"):
                return url, f"{media_type}生成成功！\n链接: {url}"
            elif url:
                return "", f"{media_type}生成成功！（base64数据，长度 {len(url)}）"
            else:
                return "", f"未获取到链接\n{json.dumps(item, indent=2, ensure_ascii=False)}"
        return "", json.dumps(result, indent=2, ensure_ascii=False)

    # ---- 模型加载 ----

    def _load_models(self):
        self.status_var.set("正在加载模型列表...")
        self.connection_var.set("加载中...")
        try:
            models = self._sync_request(self.client.list_models)
            categories = self.client.categorize_models(models)
            self._update_model_ui(categories, len(models))
        except Exception:
            self.connection_var.set("连接失败")
            self.status_var.set("就绪")

    def _update_model_ui(self, categories, count):
        chat_names = [m["name"] for m in categories["chat"]]
        image_names = [m["name"] for m in categories["image"]]
        video_names = [m["name"] for m in categories["video"]]

        self.chat_model_combo["values"] = chat_names
        self.image_model_combo["values"] = image_names
        self.video_model_combo["values"] = video_names
        self.chat_model_ids = {m["name"]: m["id"] for m in categories["chat"]}
        self.image_model_ids = {m["name"]: m["id"] for m in categories["image"]}
        self.video_model_ids = {m["name"]: m["id"] for m in categories["video"]}

        if chat_names: self.chat_model_combo.current(0)
        if image_names: self.image_model_combo.current(0)
        if video_names: self.video_model_combo.current(0)

        self.connection_var.set(f"已连接 ({count} 个模型)")
        self.status_var.set(f"模型加载完成: 聊天 {len(chat_names)} 个, 图片 {len(image_names)} 个, 视频 {len(video_names)} 个")

    def _test_connection(self):
        self.connection_var.set("测试中...")
        try:
            models = self._sync_request(self.client.list_models)
            self.connection_var.set("连接正常" if models else "连接失败")
        except Exception:
            self.connection_var.set("连接失败")

    # ---- 提交请求 ----

    def _submit_chat(self):
        if self._busy:
            return
        prompt = self.chat_prompt.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("提示", "请输入提示词")
            return

        model_id = self._get_model_id(self.chat_model_var.get(), self.chat_model_ids)
        temp = self.chat_temp_var.get()
        max_tokens = self.chat_maxtokens_var.get()
        self.status_var.set(f"正在请求 {self.chat_model_var.get()}...")

        try:
            result = self._sync_request(
                self.client.chat_completion, prompt, model_id, temp, max_tokens
            )
            self._set_result(self.chat_result, result)
        except Exception as e:
            self._set_result(self.chat_result, f"错误: {e}")
        finally:
            self.status_var.set("就绪")

    def _submit_image(self):
        if self._busy:
            return
        prompt = self.image_prompt.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("提示", "请输入提示词")
            return

        model_id = self._get_model_id(self.image_model_var.get(), self.image_model_ids)
        size = self.image_size_var.get()
        self.status_var.set("正在生成图片...")
        self._set_result(self.image_result, "正在生成图片，请稍候...")
        self.image_link_var.set("")

        try:
            result = self._sync_request(
                self.client.generate_image, prompt, model_id, size
            )
            url, msg = self._extract_url(result, "图片")
            self._set_result(self.image_result, msg)
            if url:
                self.image_link_var.set(url)
        except Exception as e:
            self._set_result(self.image_result, f"错误: {e}")
        finally:
            self.status_var.set("就绪")

    def _submit_video(self):
        if self._busy:
            return
        prompt = self.video_prompt.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("提示", "请输入提示词")
            return

        model_id = self._get_model_id(self.video_model_var.get(), self.video_model_ids)
        image_path = self.video_image_path.get() or None
        self.status_var.set("正在生成视频...")
        self._set_result(self.video_result, "正在生成视频，请稍候...")
        self.video_link_var.set("")

        try:
            result = self._sync_request(
                self.client.generate_video, prompt, model_id, image_path
            )
            url, msg = self._extract_url(result, "视频")
            self._set_result(self.video_result, msg)
            if url:
                self.video_link_var.set(url)
        except Exception as e:
            self._set_result(self.video_result, f"错误: {e}")
        finally:
            self.status_var.set("就绪")


def main():
    root = tk.Tk()
    app = APIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
