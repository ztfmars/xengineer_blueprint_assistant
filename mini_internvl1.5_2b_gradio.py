import gradio as gr
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import os
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy import pipeline, ChatTemplateConfig
from modelscope import snapshot_download


# model_path = '/home/fusionai/project/internvl/internVL_demo/train/internvl_chat_v1_5_internlm2_1_8b_logic500_ft'
snapshot_download('ztfmars/Mini-InternVL-Chat-2B-V1-5-ft',cache_dir='./llm_model')

cur_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{cur_dir}/llm_model/ztfmars/Mini-InternVL-Chat-2B-V1-5-ft"

# use lmdeploy
backend_config = TurbomindEngineConfig(session_len=8192,cache_max_entry_count=0.05) # 图片分辨率较高时请调高session_len
pipe = pipeline(model_path,
                chat_template_config=ChatTemplateConfig(model_name='internvl-internlm2'),
                backend_config=backend_config)

# use lmdeploy
backend_config = TurbomindEngineConfig(session_len=8192,cache_max_entry_count=0.05) # 图片分辨率较高时请调高session_len
pipe = pipeline(model_path,
                chat_template_config=ChatTemplateConfig(model_name='internvl-internlm2'),
                backend_config=backend_config)

# 创建模型预测函数
def build_model(image, text, temperature, top_p, max_new_tokens):

    gen_config = GenerationConfig(top_p=top_p,
                            top_k=40,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens)

    gen_config = GenerationConfig(top_p=top_p,
                            top_k=40,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens)

    if image is None:
        return [(text, "请上传一张图片。")]
    else:
        # pixel_values = load_image(image)
        response = pipe((text, image), gen_config=gen_config).text
        print("------> response: ", response)
        # pixel_values = load_image(image)
        response = pipe((text, image), gen_config=gen_config).text
        print("------> response: ", response)
        return [(text, response)]

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

# 创建 Gradio 界面
with gr.Blocks(title="InternVL-Chat", theme=gr.themes.Default(), css=block_css) as demo:
    gr.HTML("<h1 style='text-align: center; font-size: 36px;'>核电逻辑图纸识别小助手</h1>")  # 添加居中且变大的标题
    
    with gr.Row():
        with gr.Column(scale=3):
            # 输入组件
            image_input = gr.Image(type="pil", label="Upload Image")

            # 参数调节组件
            with gr.Accordion("Parameters", open=False) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="Temperature")
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P")
                max_output_tokens = gr.Slider(minimum=0, maximum=4096, value=512, step=64, interactive=True, label="Max output tokens")
            text_input = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
            # 示例
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            gr.Examples(examples=[
                [f"{cur_dir}/imgs/test0001.png", "1MCR033ST或1MCR035MT信号出现将会引起什么后果？"],
                [f"{cur_dir}/imgs/test0002.png", "这张图片主要描述了什么内容？"],
            ], inputs=[image_input, text_input])  

        with gr.Column(scale=8):
            # 输出组件
            output_chatbot = gr.Chatbot(elem_id="chatbot", label="Conversations", height=550)
            submit_btn = gr.Button(value="Send", variant="primary")
            

    # 设置回调
    submit_btn.click(
        fn=build_model, 
        inputs=[image_input, text_input, temperature, top_p, max_output_tokens], 
        outputs=output_chatbot
    )


if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port= 7860, show_error=True, share=True) 
    