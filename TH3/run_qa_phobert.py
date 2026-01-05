"""
Fine-tuning PhoBERT for Question Answering on UIT ViQuAD 2.0 dataset
Using fairseq to load model and vncorenlp for word segmentation
"""

import os
import sys
import json
import re
import logging
import argparse
import urllib.request
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Fairseq imports
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

# VnCoreNLP import
try:
    from vncorenlp import VnCoreNLP
    VNCORENLP_AVAILABLE = True
except ImportError:
    VNCORENLP_AVAILABLE = False
    print("Warning: vncorenlp not available. Word segmentation will be skipped.")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    phobert_path: str = field(
        default="PhoBERT_base_fairseq",
        metadata={"help": "Path to PhoBERT fairseq model directory"}
    )
    vncorenlp_jar: Optional[str] = field(
        default=None,
        metadata={"help": "Path to VnCoreNLP jar file"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration"""
    train_file: Optional[str] = field(default=None, metadata={"help": "Training data file (JSON)"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Validation data file (JSON)"})
    test_file: Optional[str] = field(default=None, metadata={"help": "Test data file (JSON)"})
    max_seq_length: int = field(default=256, metadata={"help": "Maximum sequence length"})
    doc_stride: int = field(default=128, metadata={"help": "Document stride"})
    max_answer_length: int = field(default=30, metadata={"help": "Maximum answer length"})
    version_2_with_negative: bool = field(default=True, metadata={"help": "Use SQuAD v2 format"})


@dataclass
class TrainingArguments:
    """Arguments for training configuration"""
    output_dir: str = field(default="./phobert_qa_output", metadata={"help": "Output directory"})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Training batch size"})
    per_device_eval_batch_size: int = field(default=16, metadata={"help": "Evaluation batch size"})
    learning_rate: float = field(default=3e-5, metadata={"help": "Learning rate"})
    warmup_steps: int = field(default=500, metadata={"help": "Warmup steps"})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X steps"})
    logging_steps: int = field(default=100, metadata={"help": "Log every X steps"})
    seed: int = field(default=42, metadata={"help": "Random seed"})


def find_or_download_vncorenlp(jar_path: Optional[str] = None) -> Optional[str]:
    """
    Tìm hoặc tải VnCoreNLP jar file tự động bằng command line
    """
    # Nếu đã có đường dẫn và file tồn tại
    if jar_path and os.path.exists(jar_path):
        logger.info(f"Found VnCoreNLP jar at: {jar_path}")
        return jar_path
    
    # Tìm trong các thư mục thông thường
    possible_paths = [
        jar_path,  # Path được chỉ định
        "vncorenlp/VnCoreNLP-1.1.1.jar",
        "./vncorenlp/VnCoreNLP-1.1.1.jar",
        "../vncorenlp/VnCoreNLP-1.1.1.jar",
        os.path.join(os.path.dirname(__file__), "vncorenlp/VnCoreNLP-1.1.1.jar"),
        os.path.expanduser("~/vncorenlp/VnCoreNLP-1.1.1.jar"),
        "/content/vncorenlp/VnCoreNLP-1.1.1.jar",  # Google Colab
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            logger.info(f"Found VnCoreNLP jar at: {path}")
            return path
    
    # Nếu không tìm thấy, tự động tải về bằng command line
    logger.info("VnCoreNLP jar not found. Downloading using command line...")
    
    vncorenlp_dir = "vncorenlp"
    models_dir = os.path.join(vncorenlp_dir, "models", "wordsegmenter")
    
    # Tạo thư mục
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Created directory: {models_dir}")
    
    # URLs
    jar_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar"
    vi_vocab_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab"
    wordsegmenter_rdr_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr"
    
    # File paths
    jar_file = os.path.join(vncorenlp_dir, "VnCoreNLP-1.1.1.jar")
    vi_vocab_file = os.path.join(models_dir, "vi-vocab")
    wordsegmenter_rdr_file = os.path.join(models_dir, "wordsegmenter.rdr")
    
    # Temporary download paths
    temp_jar = "VnCoreNLP-1.1.1.jar"
    temp_vi_vocab = "vi-vocab"
    temp_wordsegmenter_rdr = "wordsegmenter.rdr"
    
    try:
        # Kiểm tra xem có wget hay curl không
        has_wget = shutil.which("wget") is not None
        has_curl = shutil.which("curl") is not None
        
        if not has_wget and not has_curl:
            logger.error("Neither wget nor curl found. Please install one of them.")
            logger.warning("Will continue without word segmentation")
            return None
        
        # Tải VnCoreNLP-1.1.1.jar
        logger.info(f"Downloading VnCoreNLP-1.1.1.jar...")
        if has_wget:
            subprocess.run(["wget", jar_url, "-O", temp_jar], check=True)
        else:
            subprocess.run(["curl", "-L", jar_url, "-o", temp_jar], check=True)
        
        # Di chuyển jar file vào thư mục vncorenlp
        if os.path.exists(temp_jar):
            shutil.move(temp_jar, jar_file)
            logger.info(f"Downloaded and moved VnCoreNLP jar to: {jar_file}")
        
        # Tải vi-vocab
        if not os.path.exists(vi_vocab_file):
            logger.info("Downloading vi-vocab...")
            if has_wget:
                subprocess.run(["wget", vi_vocab_url, "-O", temp_vi_vocab], check=True)
            else:
                subprocess.run(["curl", "-L", vi_vocab_url, "-o", temp_vi_vocab], check=True)
            
            if os.path.exists(temp_vi_vocab):
                shutil.move(temp_vi_vocab, vi_vocab_file)
                logger.info(f"Downloaded and moved vi-vocab to: {vi_vocab_file}")
        
        # Tải wordsegmenter.rdr
        if not os.path.exists(wordsegmenter_rdr_file):
            logger.info("Downloading wordsegmenter.rdr...")
            if has_wget:
                subprocess.run(["wget", wordsegmenter_rdr_url, "-O", temp_wordsegmenter_rdr], check=True)
            else:
                subprocess.run(["curl", "-L", wordsegmenter_rdr_url, "-o", temp_wordsegmenter_rdr], check=True)
            
            if os.path.exists(temp_wordsegmenter_rdr):
                shutil.move(temp_wordsegmenter_rdr, wordsegmenter_rdr_file)
                logger.info(f"Downloaded and moved wordsegmenter.rdr to: {wordsegmenter_rdr_file}")
        
        # Kiểm tra xem jar file đã tồn tại chưa
        if os.path.exists(jar_file):
            logger.info("VnCoreNLP setup completed successfully")
            return jar_file
        else:
            logger.error("Failed to download VnCoreNLP jar file")
            return None
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download VnCoreNLP using command line: {e}")
        logger.warning("Will continue without word segmentation")
        return None
    except Exception as e:
        logger.error(f"Failed to download VnCoreNLP: {e}")
        logger.warning("Will continue without word segmentation")
        return None


class VnCoreNLPSegmenter:
    """Wrapper for VnCoreNLP word segmentation"""
    
    def __init__(self, jar_path: Optional[str] = None):
        self.segmenter = None
        self.jar_path = None
        
        if not VNCORENLP_AVAILABLE:
            logger.warning("vncorenlp package not installed. Install with: pip install vncorenlp")
            return
        
        # Tự động tìm hoặc tải jar file
        self.jar_path = find_or_download_vncorenlp(jar_path)
        
        if self.jar_path:
            try:
                self.segmenter = VnCoreNLP(self.jar_path, annotators="wseg", max_heap_size='-Xmx500m')
                logger.info("VnCoreNLP initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize VnCoreNLP: {e}")
                self.segmenter = None
        else:
            logger.warning("VnCoreNLP jar not found. Word segmentation will be skipped")
    
    def segment(self, text: str) -> str:
        """Segment Vietnamese text"""
        if not text:
            return ""
        if self.segmenter is None:
            return text
        try:
            sentences = self.segmenter.tokenize(text)
            # Join words with spaces (VnCoreNLP uses _ for compound words)
            return " ".join([" ".join(s) for s in sentences])
        except Exception as e:
            logger.warning(f"Segmentation error: {e}")
            return text


def load_json_robust(file_path: str) -> dict:
    """Load JSON file with robust error handling for invalid escape sequences"""
    logger.info(f"Loading JSON file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix invalid escape sequences
    # 1. Fix invalid \u sequences
    content = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', content)
    
    # 2. Protect valid escape sequences
    valid_escapes = {
        r'\\"': '__QUOTE_ESCAPE__',
        r'\\\\': '__BACKSLASH_ESCAPE__',
        r'\\/': '__SLASH_ESCAPE__',
        r'\\b': '__BS_ESCAPE__',
        r'\\f': '__FF_ESCAPE__',
        r'\\n': '__LF_ESCAPE__',
        r'\\r': '__CR_ESCAPE__',
        r'\\t': '__TAB_ESCAPE__',
    }
    
    protected_content = content
    for pattern, placeholder in valid_escapes.items():
        protected_content = protected_content.replace(pattern, placeholder)
    
    # 3. Escape all other backslashes
    protected_content = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', protected_content)
    
    # 4. Restore valid escapes
    for pattern, placeholder in valid_escapes.items():
        protected_content = protected_content.replace(placeholder, pattern)
    
    # 5. Handle unicode escapes
    protected_content = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: f'\\u{m.group(1)}', protected_content)
    
    # 6. Cleanup multiple backslashes
    protected_content = re.sub(r'\\\\{3,}', r'\\\\', protected_content)
    
    try:
        data = json.loads(protected_content)
        logger.info("JSON loaded successfully")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error at position {e.pos}: {e.msg}")
        raise


class QADataset(Dataset):
    """Dataset for Question Answering"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: RobertaModel,
        vocab: Dictionary,
        bpe: fastBPE,
        segmenter: VnCoreNLPSegmenter,
        max_seq_length: int = 256,
        doc_stride: int = 128,
        is_training: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.bpe = bpe
        self.segmenter = segmenter
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.is_training = is_training
        self.features = []
        
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare features from raw data"""
        logger.info(f"Preparing features (is_training={self.is_training})...")
        
        for item in tqdm(self.data, desc="Processing examples"):
            # Segment text using VnCoreNLP
            context_raw = item['context']
            question_raw = item['question']
            
            context_seg = self.segmenter.segment(context_raw)
            question_seg = self.segmenter.segment(question_raw)
            
            # Tokenize using BPE
            question_bpe = self.bpe.encode('<s> ' + question_seg + ' </s>')
            context_bpe = self.bpe.encode(context_seg)
            
            question_ids = self.vocab.encode_line(question_bpe, append_eos=False, add_if_not_exist=False).long()
            context_ids = self.vocab.encode_line(context_bpe, append_eos=False, add_if_not_exist=False).long()
            
            # Calculate available space for context
            special_tokens = 3  # <s>, </s>, </s>
            max_context_len = self.max_seq_length - len(question_ids) - special_tokens
            
            if max_context_len <= 0:
                # Truncate question if too long
                max_q_len = self.max_seq_length - special_tokens - 10
                question_ids = question_ids[:max_q_len]
                max_context_len = self.max_seq_length - len(question_ids) - special_tokens
            
            # Split context into chunks if needed
            if len(context_ids) <= max_context_len:
                chunks = [(0, len(context_ids))]
            else:
                chunks = []
                start = 0
                while start < len(context_ids):
                    end = min(start + max_context_len, len(context_ids))
                    chunks.append((start, end))
                    if end == len(context_ids):
                        break
                    start = max(start + 1, end - self.doc_stride)
            
            # Create features for each chunk
            for chunk_start, chunk_end in chunks:
                context_chunk = context_ids[chunk_start:chunk_end]
                
                # Build input: <s> question </s> context </s>
                input_ids = torch.cat([
                    torch.tensor([self.vocab.bos()]),  # <s>
                    question_ids,
                    torch.tensor([self.vocab.eos()]),  # </s>
                    context_chunk,
                    torch.tensor([self.vocab.eos()]),  # </s>
                ])
                
                # Truncate if needed
                if len(input_ids) > self.max_seq_length:
                    input_ids = input_ids[:self.max_seq_length]
                    input_ids[-1] = self.vocab.eos()
                
                # Pad if needed
                if len(input_ids) < self.max_seq_length:
                    padding = torch.full((self.max_seq_length - len(input_ids),), self.vocab.pad())
                    input_ids = torch.cat([input_ids, padding])
                
                # Create attention mask
                attention_mask = (input_ids != self.vocab.pad()).long()
                
                # Prepare labels
                start_positions = None
                end_positions = None
                
                if self.is_training and 'answers' in item and item['answers']['text']:
                    # Find answer positions
                    answer_text = item['answers']['text'][0]
                    answer_seg = self.segmenter.segment(answer_text)
                    answer_start_char = item['answers']['answer_start'][0]
                    
                    # Find answer in segmented context
                    answer_start_in_seg = context_seg.find(answer_seg)
                    
                    if answer_start_in_seg != -1:
                        # Calculate token positions
                        # This is approximate - in practice, need more precise mapping
                        question_len = len(question_ids) + 2  # +2 for <s> and </s>
                        context_start_token = question_len
                        
                        # Approximate token positions (simplified)
                        # In practice, need to map character positions to token positions
                        start_positions = context_start_token
                        end_positions = context_start_token + len(context_chunk) - 1
                
                feature = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'start_positions': start_positions,
                    'end_positions': end_positions,
                    'example_id': item['id'],
                    'context': context_seg,
                    'question': question_seg,
                }
                
                if not self.is_training:
                    feature['answers'] = item.get('answers', {'text': [], 'answer_start': []})
                
                self.features.append(feature)
        
        logger.info(f"Prepared {len(self.features)} features from {len(self.data)} examples")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class PhoBERTQA(nn.Module):
    """PhoBERT model with QA head"""
    
    def __init__(self, phobert_model: RobertaModel, num_labels: int = 2):
        super().__init__()
        self.phobert = phobert_model
        self.qa_outputs = nn.Linear(self.phobert.model.decoder.sentence_encoder.embed_tokens.embedding_dim, num_labels)
    
    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        # Get hidden states from PhoBERT
        features = self.phobert.extract_features(input_ids)
        
        # Get sequence output (batch_size, seq_len, hidden_size)
        sequence_output = features
        
        # QA head
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        return {
            'loss': total_loss,
            'start_logits': start_logits,
            'end_logits': end_logits,
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    
    if batch[0].get('start_positions') is not None:
        result['start_positions'] = torch.tensor([item['start_positions'] for item in batch])
        result['end_positions'] = torch.tensor([item['end_positions'] for item in batch])
    else:
        result['example_ids'] = [item['example_id'] for item in batch]
        result['contexts'] = [item['context'] for item in batch]
        result['questions'] = [item['question'] for item in batch]
        if 'answers' in batch[0]:
            result['answers'] = [item['answers'] for item in batch]
    
    return result


def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--phobert_path", type=str, default="PhoBERT_base_fairseq",
                       help="Path to PhoBERT fairseq model directory")
    parser.add_argument("--vncorenlp_jar", type=str, default=None,
                       help="Path to VnCoreNLP jar file (if not specified, will auto-download)")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default=None, help="Training data file")
    parser.add_argument("--validation_file", type=str, default=None, help="Validation data file")
    parser.add_argument("--test_file", type=str, default=None, help="Test data file")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--doc_stride", type=int, default=128, help="Document stride")
    parser.add_argument("--max_answer_length", type=int, default=30, help="Maximum answer length")
    parser.add_argument("--n_best_size", type=int, default=20, help="Number of n-best predictions to generate")
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0, help="Threshold for null answer prediction")
    parser.add_argument("--version_2_with_negative", action="store_true", help="Use SQuAD v2 format")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./phobert_qa_output", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load PhoBERT model from fairseq
    logger.info(f"Loading PhoBERT from {args.phobert_path}")
    phobert = RobertaModel.from_pretrained(args.phobert_path, checkpoint_file='model.pt')
    phobert.eval()  # Set to train mode later if needed
    
    # Load BPE
    class BPE:
        bpe_codes = os.path.join(args.phobert_path, 'bpe.codes')
    
    bpe_args = BPE()
    phobert.bpe = fastBPE(bpe_args)
    
    # Load dictionary
    vocab = Dictionary()
    dict_path = os.path.join(args.phobert_path, 'dict.txt')
    vocab.add_from_file(dict_path)
    
    logger.info(f"PhoBERT loaded. Vocab size: {len(vocab)}")
    
    # Initialize VnCoreNLP segmenter
    segmenter = VnCoreNLPSegmenter(args.vncorenlp_jar)
    
    # Load datasets
    train_data = []
    val_data = []
    test_data = []
    
    if args.train_file:
        train_json = load_json_robust(args.train_file)
        train_data = train_json.get('data', train_json) if isinstance(train_json, dict) else train_json
        logger.info(f"📊 Loaded {len(train_data)} training examples")
    
    if args.validation_file:
        val_json = load_json_robust(args.validation_file)
        val_data = val_json.get('data', val_json) if isinstance(val_json, dict) else val_json
        logger.info(f"📊 Loaded {len(val_data)} validation examples")
    
    if args.test_file:
        test_json = load_json_robust(args.test_file)
        test_data = test_json.get('data', test_json) if isinstance(test_json, dict) else test_json
        logger.info(f"📊 Loaded {len(test_data)} test examples")
    
    # Create datasets
    train_dataset = None
    val_dataset = None
    test_dataset = None
    
    if args.do_train and train_data:
        train_dataset = QADataset(
            train_data, phobert, vocab, phobert.bpe, segmenter,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            is_training=True,
        )
    
    if args.do_eval and val_data:
        val_dataset = QADataset(
            val_data, phobert, vocab, phobert.bpe, segmenter,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            is_training=False,
        )
    
    if args.do_predict and test_data:
        test_dataset = QADataset(
            test_data, phobert, vocab, phobert.bpe, segmenter,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            is_training=False,
        )
    
    # Create model
    model = PhoBERTQA(phobert)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"Model created and moved to {device}")
    
    # Training
    if args.do_train and train_dataset:
        logger.info("Starting training...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        model.train()
        phobert.train()  # Enable training mode
        
        global_step = 0
        for epoch in range(args.num_train_epochs):
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % args.logging_steps == 0:
                    logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")
                
                if global_step % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            logger.info(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/len(train_loader):.4f}")
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(final_model_path, "pytorch_model.bin"))
        logger.info(f"Saved final model to {final_model_path}")
    
    # Evaluation and prediction
    if args.do_eval or args.do_predict:
        model.eval()
        phobert.eval()
        
        # Import evaluation functions
        try:
            from evaluation import (
                normalize_answer, compute_exact, compute_f1, 
                get_raw_scores, make_eval_dict, make_qid_to_has_ans
            )
            EVALUATION_AVAILABLE = True
        except ImportError:
            logger.warning("evaluation.py not found. Will skip detailed evaluation metrics.")
            EVALUATION_AVAILABLE = False
        
        for dataset_name, dataset in [("validation", val_dataset), ("test", test_dataset)]:
            if dataset is None:
                continue
            
            logger.info(f"Running {dataset_name}...")
            loader = DataLoader(
                dataset,
                batch_size=args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            
            # Collect all predictions with logits for post-processing
            all_start_logits = []
            all_end_logits = []
            all_example_ids = []
            all_contexts = []
            all_questions = []
            all_answers = []
            
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    start_logits = outputs['start_logits'].cpu().numpy()
                    end_logits = outputs['end_logits'].cpu().numpy()
                    
                    all_start_logits.append(start_logits)
                    all_end_logits.append(end_logits)
                    all_example_ids.extend(batch['example_ids'])
                    all_contexts.extend(batch['contexts'])
                    all_questions.extend(batch['questions'])
                    if 'answers' in batch:
                        all_answers.extend(batch['answers'])
            
            # Concatenate all logits
            all_start_logits = np.concatenate(all_start_logits, axis=0)
            all_end_logits = np.concatenate(all_end_logits, axis=0)
            
            # Post-process predictions
            # Group features by example_id (since one example can have multiple features)
            example_to_features = {}
            for idx, example_id in enumerate(all_example_ids):
                if example_id not in example_to_features:
                    example_to_features[example_id] = {
                        'features': [],
                        'context': all_contexts[idx],
                        'question': all_questions[idx],
                        'answers': all_answers[idx] if idx < len(all_answers) else {'text': [], 'answer_start': []}
                    }
                example_to_features[example_id]['features'].append({
                    'start_logits': all_start_logits[idx],
                    'end_logits': all_end_logits[idx],
                    'feature_idx': idx
                })
            
            # Extract answers for each example
            predictions_dict = {}
            n_best_size = 20
            
            for example_id, example_data in tqdm(example_to_features.items(), desc="Post-processing predictions"):
                context = example_data['context']
                features = example_data['features']
                
                # Get best prediction across all features for this example
                prelim_predictions = []
                min_null_score = None
                
                for feature in features:
                    start_logits = feature['start_logits']
                    end_logits = feature['end_logits']
                    
                    # Null score (CLS token)
                    null_score = start_logits[0] + end_logits[0]
                    if min_null_score is None or null_score < min_null_score:
                        min_null_score = null_score
                    
                    # Get top n_best_size start and end positions
                    start_indexes = np.argsort(start_logits)[-1:-n_best_size-1:-1].tolist()
                    end_indexes = np.argsort(end_logits)[-1:-n_best_size-1:-1].tolist()
                    
                    for start_idx in start_indexes:
                        for end_idx in end_indexes:
                            if end_idx < start_idx or end_idx - start_idx + 1 > args.max_answer_length:
                                continue
                            
                            # Approximate: extract text from context based on token positions
                            # This is simplified - in practice need proper offset mapping
                            score = start_logits[start_idx] + end_logits[end_idx]
                            prelim_predictions.append({
                                'start_idx': start_idx,
                                'end_idx': end_idx,
                                'score': score,
                                'start_logit': start_logits[start_idx],
                                'end_logit': end_logits[end_idx],
                            })
                
                # Sort by score and take best
                prelim_predictions = sorted(prelim_predictions, key=lambda x: x['score'], reverse=True)[:n_best_size]
                
                # Extract answer text (simplified - using token indices)
                # In practice, need proper offset mapping from tokens to characters
                if len(prelim_predictions) > 0:
                    best_pred = prelim_predictions[0]
                    # For now, use a simple heuristic: decode tokens
                    # This is approximate and should be improved with proper offset mapping
                    pred_text = ""  # Will be improved with proper token-to-text mapping
                    
                    # Simple fallback: if we can't extract properly, use empty string
                    if args.version_2_with_negative and min_null_score is not None:
                        best_score = best_pred['score']
                        if best_score < min_null_score - args.null_score_diff_threshold:
                            pred_text = ""
                        else:
                            # Try to extract from context (simplified)
                            # TODO: Implement proper token-to-character mapping
                            pred_text = ""
                else:
                    pred_text = ""
                
                predictions_dict[example_id] = pred_text
            
            # Save predictions in format compatible with evaluation.py
            predictions_file = os.path.join(args.output_dir, f"{dataset_name}_predictions.json")
            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(predictions_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved predictions to {predictions_file}")
            
            # Run evaluation if we have ground truth
            if args.do_eval and dataset_name == "validation" and EVALUATION_AVAILABLE:
                # Prepare data in format expected by evaluation.py
                eval_data = []
                for example_id, example_data in example_to_features.items():
                    eval_data.append({
                        'id': example_id,
                        'context': example_data['context'],
                        'question': example_data['question'],
                        'answers': example_data['answers']
                    })
                
                # Convert to SQuAD format for evaluation
                squad_format_data = {
                    'data': [{
                        'paragraphs': [{
                            'context': eval_data[0]['context'] if eval_data else '',
                            'qas': [{
                                'id': item['id'],
                                'question': item['question'],
                                'answers': item['answers']['text'] if item['answers']['text'] else []
                            } for item in eval_data]
                        }]
                    }]
                }
                
                # Calculate metrics
                qid_to_has_ans = make_qid_to_has_ans(squad_format_data['data'])
                exact_scores, f1_scores = get_raw_scores(squad_format_data['data'], predictions_dict)
                eval_dict = make_eval_dict(exact_scores, f1_scores)
                
                # Add HasAns/NoAns breakdown
                has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
                no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
                
                if has_ans_qids:
                    has_ans_eval = make_eval_dict(exact_scores, f1_scores, qid_list=has_ans_qids)
                    for k, v in has_ans_eval.items():
                        eval_dict[f'HasAns_{k}'] = v
                
                if no_ans_qids:
                    no_ans_eval = make_eval_dict(exact_scores, f1_scores, qid_list=no_ans_qids)
                    for k, v in no_ans_eval.items():
                        eval_dict[f'NoAns_{k}'] = v
                
                # Save evaluation results
                eval_results_file = os.path.join(args.output_dir, f"{dataset_name}_eval_results.json")
                with open(eval_results_file, 'w', encoding='utf-8') as f:
                    json.dump(eval_dict, f, indent=2)
                
                logger.info(f"Evaluation results for {dataset_name}:")
                logger.info(f"  Exact Match: {eval_dict.get('exact', 0):.2f}%")
                logger.info(f"  F1: {eval_dict.get('f1', 0):.2f}%")
                if has_ans_qids:
                    logger.info(f"  HasAns Exact: {eval_dict.get('HasAns_exact', 0):.2f}%")
                    logger.info(f"  HasAns F1: {eval_dict.get('HasAns_f1', 0):.2f}%")
                if no_ans_qids:
                    logger.info(f"  NoAns Exact: {eval_dict.get('NoAns_exact', 0):.2f}%")
                    logger.info(f"  NoAns F1: {eval_dict.get('NoAns_f1', 0):.2f}%")
                logger.info(f"Saved evaluation results to {eval_results_file}")
            
            # Optionally run evaluation.py script
            if args.do_predict and dataset_name == "test":
                logger.info(f"Predictions saved. You can run evaluation.py separately:")
                logger.info(f"  python evaluation.py {args.test_file} {predictions_file}")


if __name__ == "__main__":
    main()