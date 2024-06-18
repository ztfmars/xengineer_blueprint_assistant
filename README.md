## 说明
传统的单一模态(如仅使用图像)的图纸分析无法充分利用多源信息，导致识别与理解存在局限。针对此现状，我们开发出 Nuclear Blueprint Assistant-核电工程图纸识别的视觉大语言模型。

该系统围绕核电领域，结合工程专家和gpt标注相结合，并且利于InternLLm20B进行语义扩展和形式扩充，极大增加了数据多样性和泛化效果。
该系统综合对比xcomposer2-4khd/internVL1.5/llava-llama3-8b模型采用最优模型，结合核电工程图纸与相关文档数据，通过多模态数据的有机结合，挖掘出隐藏的关联模式，提升了识别与分析的效果，打破了传统单一模态分析的局限。
在数据、业务保密，外界不能访问垂直行业，对于特殊表格符号含义、增强逻辑推理判断、few-shot提升模型泛化效果等方面进行了积极探索。

该系统在以下领域具备显著应用前景： 
- 工程图纸识别和关键信息提取、理解
- 工业图纸条件逻辑推理和判断，辅助决策和优化应急方法
- 工程图纸审核校验
- 核电行业垂直行业知识问答
- 核工业知识培训和教育
该系统的部署与应用能够显著提升核电工程图纸的处理效率与准确性，推动核电领域智能化水平的提升。
## 架构图
![架构图](https://github.com/ztfmars/xengineer_blueprint_assistant/assets/35566228/13c04481-507d-46f7-bddf-834062b85394)

## 效果图
![效果图](https://github.com/ztfmars/xengineer_blueprint_assistant/assets/35566228/3fece665-8dab-46ac-a027-34fd9a12f223)

## b站视频
https://www.bilibili.com/video/BV1f1421k7r4?p=1&vd_source=6a2cd0e00818aa6143834cc965402bca
