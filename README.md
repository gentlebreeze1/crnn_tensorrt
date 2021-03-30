
# train
  crnn训练首先要生成一个密码本，包含数据集中所有的汉字
  然后生产train.txt和test.txt格式如下：文件名 汉字所对应密码本的索引
  
  ![image](https://user-images.githubusercontent.com/27668596/112813126-e098f680-904b-11eb-9688-5fca38144b67.png)
  
  执行 python3 train.py就可以训练
  # onnx及tensorrt导出及推理
  tensorrt的版本是7.2.3
  运行export.py既可以导出onnx模型
  cd到trt目录  执行convert_trt_quant.py既可以生成tensorrt模型
  执行demo_trt既可以获取推理结果
  推理时注意tensorrt的input要和onnx模型的input shape的对其
