#!/bin/bash

# 设置默认最大执行数量
MAX_PROCESSES=10
DRYRUN=false

# 解析输入参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-processes)
      MAX_PROCESSES="$2"
      shift # past argument
      shift # past value
      ;;
    --dryrun)
      DRYRUN=true
      shift # past argument
      ;;
    *)
      shift # past unrecognized argument
      ;;
  esac
done

# 设置基础路径
LLAMA_FACTORY_PATH="/home/zhangji/LLM/LLaMA-Factory_AI4S"

# 设置模型和脚本路径
MODEL_PATH="$LLAMA_FACTORY_PATH/pretrain_ckp/dbg_gemma1.3"
TOKENIZER_SCRIPT="$LLAMA_FACTORY_PATH/src/tokenizer.py"

# 其他参数
PREPROCESSING_NUM_WORKERS=2
SEED=1
STAGE="pt"
STREAMING="true"
MAX_STEPS=1000000000

# 获取当前目录
CURRENT_DIR=$(pwd)

# 设置输出路径为当前目录下的 tokenized_data 子目录
TOKENIZED_DIR="$CURRENT_DIR/tokenized_data"
OUTPUT_DIR="$CURRENT_DIR"

# 设置要统计和处理的文件扩展名
FILE_EXTENSIONS=("json" "arrow" "csv" "jsonl" "parquet" "txt")

# 检测 tokenized_data 目录是否存在
if [ -d "$TOKENIZED_DIR" ]; then
  read -p "The 'tokenized_data' directory already exists. Do you want to delete it and continue? (default: y/n): " confirm_delete
  if [ -z "$confirm_delete" ] || [ "$confirm_delete" = "y" ]; then
    echo "Deleting 'tokenized_data' directory."
    rm -rf "$TOKENIZED_DIR"
  else
    echo "Exiting script."
    exit 1
  fi
fi

# 统计当前目录下具有指定扩展名的文件数量
file_count=0
for ext in "${FILE_EXTENSIONS[@]}"; do
  count=$(find "$CURRENT_DIR" -maxdepth 1 -type f -name "*.$ext" | wc -l)
  file_count=$((file_count + count))
done

# 如果最大执行数量大于文件数量减一，则将最大执行数量限制为文件数量减一
if [ "$MAX_PROCESSES" -gt "$((file_count - 1))" ]; then
  MAX_PROCESSES="$((file_count - 1))"
fi

echo "Total files with specified extensions: $file_count"
echo "Max processes to run: $MAX_PROCESSES"

# 定义生成命令的函数
generate_command() {
  file="$1"
  echo python "$TOKENIZER_SCRIPT" \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_file "$file" \
    --tokenized_path "$TOKENIZED_DIR" \
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --stage "$STAGE" \
    --streaming "$STREAMING" \
    --max_steps "$MAX_STEPS"
}

# 导出函数和变量以便 xargs 使用
export -f generate_command
export TOKENIZER_SCRIPT MODEL_PATH CURRENT_DIR PREPROCESSING_NUM_WORKERS OUTPUT_DIR SEED STAGE STREAMING MAX_STEPS

# 遍历当前目录下具有指定扩展名的文件并并行执行命令，限制最多同时执行 MAX_PROCESSES 个 Python 程序
for ext in "${FILE_EXTENSIONS[@]}"; do
  find "$CURRENT_DIR" -maxdepth 1 -type f -name "*.$ext" | while read file; do
    # 获取文件名
    filename=$(basename "$file")

    # 判断文件名是否为 dataset_info.json，如果是则跳过
    if [ "$filename" = "dataset_info.json" ]; then
      echo "Skipping dataset_info.json"
      continue
    fi

    # 生成命令
    CMD=$(generate_command "$file")
    echo "Generated command: $CMD"
    
    if [ "$DRYRUN" = false ]; then
      # 执行命令
      eval $CMD &
    
      # 控制并行进程数量
      running_processes=$(jobs -p | wc -l)
      if [ $running_processes -ge $MAX_PROCESSES ]; then
        wait -n
      fi
    fi
  done
done

# 等待所有后台进程完成
if [ "$DRYRUN" = false ]; then
  wait
fi

