import argparse
import json
import os
from typing import Dict, List

from vllm import LLM, SamplingParams

from logger import setup_logger


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
        "Bạn là trợ lý trả lời câu hỏi đọc hiểu tiếng Việt. "
        "Trả lời ngắn gọn, trích đúng ý trong đoạn văn. Nếu không tìm thấy, trả lời \"Không rõ\".\n\n"
        "Ví dụ 1:\n"
        "Bối cảnh: \"Tháng 3 năm 1991, Jackson gia hạn hợp đồng cùng hãng Sony ... Dangerous đạt 7 lần chứng nhận đĩa Bạch kim tại Hoa Kỳ ...\"\n"
        "Câu hỏi: \"Dangerous đã mang lại những thành công gì cho Jackson?\"\n"
        "Trả lời: \"7 lần chứng nhận đĩa Bạch kim tại Hoa Kỳ và 30 triệu bản toàn cầu\"\n\n"
        "Ví dụ 2:\n"
        "Bối cảnh: \"Ireland có bốn người đoạt giải Nobel văn học là George Bernard Shaw ... James Joyce được nhìn nhận phổ biến là một trong các nhà văn quan trọng nhất của thế kỷ XX.\"\n"
        "Câu hỏi: \"Ai được nhìn nhận phổ biến là một trong các nhà văn quan trọng nhất của thế kỷ XX?\"\n"
        "Trả lời: \"James Joyce\"\n\n"
    )


def zero_shot_prefix() -> str:
    return (
        "Bạn là trợ lý trả lời câu hỏi đọc hiểu tiếng Việt. "
        "Trả lời ngắn gọn bằng tiếng Việt, ưu tiên trích đúng cụm từ trong đoạn văn. "
        "Nếu không rõ, trả lời \"Không rõ\".\n\n"
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
        build_prompt(ex, mode=mode, max_context_chars=max_context_chars) for ex in examples
    ]
    predictions: Dict[str, str] = {}

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_examples = examples[i : i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for ex, out in zip(batch_examples, outputs):
            text = out.outputs[0].text.strip()
            predictions[ex["id"]] = text
    return predictions


def main():
    parser = argparse.ArgumentParser(description="LLaMA inference with vLLM (Vietnamese QA)")
    parser.add_argument("--input_file", type=str, required=True, help="Path to Private_Test_ref.json")
    parser.add_argument("--output_file", type=str, default="./predictions_llama.json")
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
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger("llama_infer", log_dir=args.log_dir)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

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