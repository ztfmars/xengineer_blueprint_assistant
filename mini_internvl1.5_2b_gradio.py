import gradio as gr
from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import os
import logging.handlers
import sys
import datetime
import time
import json
from modelscope import snapshot_download


############## parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# model_path = '/home/fusionai/project/internvl/internVL_demo/train/internvl_chat_v1_5_internlm2_1_8b_logic500_ft'

snapshot_download('ztfmars/Mini-InternVL-Chat-2B-V1-5-ft',cache_dir='./llm_model')

cur_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{cur_dir}/llm_model/ztfmars/Mini-InternVL-Chat-2B-V1-5-ft"


############## models
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    # image = Image.open(image_file).convert('RGB')
    image = image_file.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)




############## web components

####  快速版
# generation_config = dict(
#     num_beams=1,
#     max_new_tokens=512,
#     do_sample=True,
#     temperature = 0.1,
#     top_p = 0.9
# )

# def build_model(image, text):
#     if image is None:
#         return [(text, "请上传一张图片。")]
#     else:
#         pixel_values = load_image(image, max_num=6).to(torch.bfloat16).cuda()
#         response = model.chat(tokenizer, pixel_values, text, generation_config)
        
#         return [(text, response)]

# demo = gr.Interface(fn=build_model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
# demo.queue().launch(server_name='0.0.0.0', server_port= 6006, show_error=True, share=True) 


#### 简化版
# 创建模型预测函数
def build_model(image, text, temperature, top_p, max_new_tokens):
    do_sample = True if temperature > 0.1 else False
    generation_config = dict(
        num_beams=1,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p
    )

    if image is None:
        return [(text, "请上传一张图片。")]
    else:
        pixel_values = load_image(image, max_num=6).to(torch.bfloat16).cuda()
        response = model.chat(tokenizer, pixel_values, text, generation_config)
        
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
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.1, interactive=True, label="Temperature")
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
    demo.queue().launch(server_name='0.0.0.0', server_port= 7860, show_error=True, share=True) 
    