from datasets import load_dataset, Dataset

def load_and_filter_json(path):
  print("process file named %s" % path)
  dataset = load_dataset('json', data_files=path, split="train", streaming=True)
  filtered_data = []
  
  for idx, row in enumerate(dataset):
      try:
          # 尝试处理行
          filtered_data.append(row)
        #   print(idx)
      except Exception as e:
          # 打印错误信息并继续
          print(f"Error processing row {idx}: {e}")
          continue
  filtered_dataset = Dataset.from_list(filtered_data) 
  return filtered_dataset

# 假设你的json文件路径是 'data.json'
path = "/data/zhangji/datasets/AlgebraicStack/data1/algebraic-stack-train-0006.json111"
# path = "/data/zhangji/datasets/AlgebraicStack/data1/dbg.json"
filtered_data = load_and_filter_json(path)

# # 打印处理后的数据
# for item in filtered_data:
#     print(item)
print(len(filtered_data))
