# 导入 os 模块，用于处理文件和目录路径、创建目录等操作
import os
# 导入 pdfplumber 库，用于从 PDF 文件中提取文本
import pdfplumber

# 定义一个函数 pdf_to_txt，用于将指定输入文件夹中的 PDF 文件转换为 TXT 文件并保存到输出文件夹
def pdf_to_txt(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建该文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件和文件夹
    for filename in os.listdir(input_folder):
        # 检查当前文件是否为 PDF 文件
        if filename.endswith(".pdf"):
            # 构建完整的 PDF 文件路径
            pdf_path = os.path.join(input_folder, filename)
            # 获取文件名（不包含扩展名），并添加 .txt 扩展名
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            # 构建完整的 TXT 文件路径
            txt_path = os.path.join(output_folder, txt_filename)
            
            # 使用 pdfplumber 打开 PDF 文件
            with pdfplumber.open(pdf_path) as pdf:
                # 遍历 PDF 文件的每一页，提取文本并使用换行符连接成一个字符串
                text = "\n".join(page.extract_text() for page in pdf.pages)
                # 以写入模式打开 TXT 文件，指定编码为 UTF-8
                with open(txt_path, "w", encoding="utf-8") as f:
                    # 将提取的文本写入 TXT 文件
                    f.write(text)
            # 打印转换完成的信息，显示已转换的 PDF 文件名
            print(f"转换完成: {filename}")


# 获取当前脚本文件所在的绝对路径
current_script_path = os.path.abspath(os.path.dirname(__file__))
# 打印当前脚本文件所在的路径（这里注释掉了打印语句，可根据需要取消注释进行调试）
# print(f"current_script_path: {current_script_path}")
# 构建项目数据目录的路径，通过当前脚本路径向上两级目录找到 data 目录
base_dir = os.path.join(current_script_path, '..', '..', 'data')
# 打印数据目录的路径（这里注释掉了打印语句，可根据需要取消注释进行调试）
# print(base_dir)
# 构建输入文件夹的路径，该文件夹包含原始的 PDF 报告文件
input_dir = os.path.join(base_dir, 'raw', 'reports')
# 构建输出文件夹的路径，用于保存转换后的 TXT 文件
output_dir = os.path.join(base_dir, 'processed', 'reports', 'txtfile')
# 调用 pdf_to_txt 函数，将输入文件夹中的 PDF 文件转换为 TXT 文件并保存到输出文件夹
pdf_to_txt(input_dir, output_dir)