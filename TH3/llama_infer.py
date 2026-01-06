import argparse
import json
import os
from typing import Dict, List
import sys

from vllm import LLM, SamplingParams

# sys.path.append("/content")
sys.path.append("/kaggle/working")
# sys.path.append("/kaggle/input")
from logger import setup_logger
from tqdm import tqdm


def load_examples(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    for item in data.get("data", []):
        examples.append(
            {
                "id": item["id"],
                "context": item["context"],
                "question": item["question"],
            }
        )
    return examples


def few_shot_prefix() -> str:
    return (
        "Bạn là trợ lý trả lời câu hỏi đọc hiểu tiếng Việt.\n"
        "NHIỆM VỤ: Trả lời câu hỏi CHỈ dựa trên \"Bối cảnh\".\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1) Chỉ dùng thông tin xuất hiện trong Bối cảnh. KHÔNG kiến thức ngoài, KHÔNG suy luận, KHÔNG đoán.\n"
        "2) Nếu không có câu/đoạn nào trong Bối cảnh trả lời trực tiếp => trả lời đúng 2 từ: \"Không rõ\".\n"
        "3) Trả lời ngắn gọn, đúng trọng tâm. Ưu tiên trích đúng cụm từ trong Bối cảnh.\n"
        "4) Không thêm chi tiết, không giải thích dài, không thêm ví dụ.\n"
        "5) Nếu câu hỏi yêu cầu con số/tên riêng/mốc thời gian: phải trùng khớp nguyên văn.\n\n"
        "Ví dụ 1 (trích đúng ý, không thêm):\n"
        "Bối cảnh: \"Tháng 3 năm 1991, Jackson gia hạn hợp đồng cùng hãng Sony ... Dangerous đạt 7 lần chứng nhận đĩa Bạch kim tại Hoa Kỳ ... và bán được 30 triệu bản toàn cầu.\"\n"
        "Câu hỏi: \"Dangerous đã mang lại những thành công gì cho Jackson?\"\n"
        "Trả lời: \"7 lần chứng nhận đĩa Bạch kim tại Hoa Kỳ và bán được 30 triệu bản toàn cầu\"\n\n"
        "Ví dụ 2 (tên riêng phải đúng):\n"
        "Bối cảnh: \"Ireland có bốn người đoạt giải Nobel văn học là George Bernard Shaw ... James Joyce được nhìn nhận phổ biến là một trong các nhà văn quan trọng nhất của thế kỷ XX.\"\n"
        "Câu hỏi: \"Ai được nhìn nhận phổ biến là một trong các nhà văn quan trọng nhất của thế kỷ XX?\"\n"
        "Trả lời: \"James Joyce\"\n\n"
        "Ví dụ 3 (thiếu thông tin => Không rõ):\n"
        "Bối cảnh: \"Cuốn sách được xuất bản vào năm 2012 và nhanh chóng bán chạy.\"\n"
        "Câu hỏi: \"Tác giả của cuốn sách là ai?\"\n"
        "Trả lời: \"Không rõ\"\n\n"
        "Ví dụ 4 (không suy diễn dù nghe có vẻ hợp lý):\n"
        "Bối cảnh: \"Trận đấu diễn ra tại sân vận động Mỹ Đình.\"\n"
        "Câu hỏi: \"Đội nào đã giành chiến thắng?\"\n"
        "Trả lời: \"Không rõ\"\n\n"
        "ĐỊNH DẠNG TRẢ LỜI:\n"
        "- Chỉ 1 dòng câu trả lời.\n"
        "- Nếu không có bằng chứng trực tiếp: chỉ viết \"Không rõ\".\n\n"
    )


def zero_shot_prefix() -> str:
    return (
        "Bạn là trợ lý trả lời câu hỏi đọc hiểu tiếng Việt.\n"
        "NHIỆM VỤ: Trả lời câu hỏi CHỈ dựa trên \"Bối cảnh\" được cung cấp.\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1) Tuyệt đối KHÔNG dùng kiến thức bên ngoài. Không suy luận, không đoán.\n"
        "2) Nếu trong Bối cảnh KHÔNG có thông tin để trả lời trực tiếp, hoặc không chắc chắn => trả lời đúng 2 từ: \"Không rõ\".\n"
        "3) Trả lời NGẮN GỌN, đúng trọng tâm câu hỏi. Ưu tiên trích nguyên văn cụm từ/đoạn ngắn từ Bối cảnh.\n"
        "4) Không thêm giải thích dài dòng. Không nêu lý do. Không thêm chi tiết không có trong Bối cảnh.\n"
        "5) Nếu câu hỏi có nhiều ý, chỉ trả lời những ý có bằng chứng rõ ràng trong Bối cảnh; phần còn lại => \"Không rõ\".\n"
        "6) Nếu Bối cảnh có nhiều mốc thời gian/đối tượng dễ nhầm, chỉ chọn phương án khớp đúng câu hỏi.\n\n"
        "ĐỊNH DẠNG TRẢ LỜI:\n"
        "- Chỉ viết 1 dòng câu trả lời.\n"
        "- Không viết \"Theo bối cảnh\", không gạch đầu dòng, không thêm dấu ngoặc.\n\n"
    )


def build_prompt(example: Dict, mode: str, max_context_chars: int) -> str:
    prefix = few_shot_prefix() if mode == "few-shot" else zero_shot_prefix()
    context = example["context"]
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + " ..."
    return (
        f"{prefix}"
        f"Bối cảnh: \"{context}\"\n"
        f"Câu hỏi: \"{example['question']}\"\n"
        "Trả lời:"
    )


def generate_predictions(
    llm: LLM,
    examples: List[Dict],
    mode: str,
    max_context_chars: int,
    sampling_params: SamplingParams,
    batch_size: int,
) -> Dict[str, str]:
    prompts = [
        build_prompt(ex, mode=mode, max_context_chars=max_context_chars)
        for ex in examples
    ]
    predictions: Dict[str, str] = {}

    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(prompts), batch_size),
        total=total_batches,
        desc="Generating",
        ncols=100,
    ):
        batch_prompts = prompts[i : i + batch_size]
        batch_examples = examples[i : i + batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)

        for ex, out in zip(batch_examples, outputs):
            text = out.outputs[0].text.strip()
            predictions[ex["id"]] = text

    return predictions


def main():
    parser = argparse.ArgumentParser(description="LLaMA inference with vLLM (Vietnamese QA)")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="./dataset/Private_Test_ref.json",
        help="Path to Private_Test_ref.json"
    )
    parser.add_argument("--output_file", type=str, default="./predictions/predictions_llama.json")
    parser.add_argument(
        "--model_name",
        type=str,
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        help="HF model name for LLaMA",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero-shot", "few-shot"],
        default="zero-shot",
        help="Prompt mode",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_context_chars", type=int, default=3500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save log files")
    parser.add_argument("--max_model_len", type=int, default=8192)

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger("llama_infer", log_dir=args.log_dir)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger.info(f"Loading examples from {args.input_file}")
    examples = load_examples(args.input_file)
    logger.info(f"Loaded {len(examples)} examples")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["</s>"],
    )

    logger.info(f"Initializing vLLM for {args.model_name} (AWQ 4-bit)")
    logger.info(f"  Tensor parallel size: {args.tensor_parallel_size}")
    logger.info(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max new tokens: {args.max_new_tokens}")
    logger.info(f"  Temperature: {args.temperature}, Top-p: {args.top_p}")
    
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="float16",
        quantization="awq",
        max_model_len=args.max_model_len,
    )

    logger.info(f"Generating predictions with mode={args.mode}")
    predictions = generate_predictions(
        llm=llm,
        examples=examples,
        mode=args.mode,
        max_context_chars=args.max_context_chars,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
    )

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(predictions)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()