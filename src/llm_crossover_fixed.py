"""LLM crossover with the chunk-index fix (for the matched MESB vs fixed-grid runs).

Same command line, prompts, helpers and random draws as ``src/llm_crossover.py``
(which is not modified). The only difference: the original enumerates
``parts_x[1:]`` from 0 and then writes the LLM's answer to ``parts_x[augment_idx]``,
one chunk *before* the chunk it showed the LLM. For XQC genes that overwrites
the ``ALGO = "xqc"`` header, so every crossover child fails the trainer's check
(31 of 31 invalid genes in Shazeb's xqc_island_run2 were crossover children).
Here the answer is written back to the chunk that was shown.

``generate_augmented_code`` receives the same index argument as in the
original (the 0-based position among ``parts_x[1:]``).

    uv run python src/llm_crossover_fixed.py X.py Y.py OUT.py --top_p 0.1 \\
        --temperature 0.07 --apply_quality_control False --llm_model llama3
"""

import argparse
import os
import random
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
for path in (SRC_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import llm_crossover  # noqa: E402  (the team's module, unchanged)
from cfg.constants import ROOT_DIR  # noqa: E402
from llm_utils import extract_note, generate_augmented_code, split_file, str2bool  # noqa: E402
from utils.print_utils import box_print  # noqa: E402


def augment_network_fixed(input_filename_x, input_filename_y, output_filename,
                          top_p=0.15, llm_model=None, temperature=0.1,
                          apply_quality_control=False, template_txt=None):
    parts_x = split_file(input_filename_x)
    parts_y = split_file(input_filename_y)
    # Same candidates and shuffle as the original, but keep the real chunk index.
    parts = [(x, y, idx) for idx, (x, y) in enumerate(zip(parts_x[1:], parts_y[1:]), start=1)]
    random.shuffle(parts)
    for x, y, chunk_idx in parts:
        if x.strip() != y.strip():
            break

    if template_txt:
        # A prompt written for this child (the directional emitter's): two
        # literal "{}" code slots, x's chunk then y's. Literal replacement,
        # since the prompt quotes numbers that str.format could misread.
        with open(template_txt, 'r') as file:
            template_txt = file.read()
        txt2llm = template_txt.replace("{}", x.strip(), 1).replace("{}", y.strip(), 1)
    else:
        template_fname = random.choice(['crossover.txt', 'crossover_s.txt'])
        with open(f'{ROOT_DIR}/templates/CrossOver/{template_fname}', 'r') as file:
            template_txt = file.read()
        txt2llm = template_txt.format(x.strip(), y.strip())
    code_from_llm = generate_augmented_code(txt2llm, chunk_idx - 1, apply_quality_control,
                                            top_p, llm_model, temperature)
    if not code_from_llm:
        code_from_llm = txt2llm

    note_txt = extract_note(parts_x[chunk_idx])
    parts_x[chunk_idx] = f"\n{note_txt}{code_from_llm}\n"
    llm_crossover.write_augmented_code(output_filename, parts_x, parts_y)
    box_print(f"Python code saved to {os.path.basename(output_filename)}",
              print_bbox_len=120, new_line_end=False)
    print('Job done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment Python Network Script (chunk-index fix).')
    parser.add_argument('input_filename_x', type=str)
    parser.add_argument('input_filename_y', type=str)
    parser.add_argument('output_filename', type=str)
    parser.add_argument('--llm_model', type=str, default=False)
    parser.add_argument('--top_p', type=float, default=0.15)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--apply_quality_control', type=str2bool, default=False)
    parser.add_argument('--template_txt', type=str, default=None,
                        help='Prompt file to use instead of a generic crossover prompt')
    args = parser.parse_args()
    augment_network_fixed(input_filename_x=args.input_filename_x,
                          input_filename_y=args.input_filename_y,
                          output_filename=args.output_filename,
                          top_p=args.top_p, llm_model=args.llm_model,
                          temperature=args.temperature,
                          apply_quality_control=args.apply_quality_control,
                          template_txt=args.template_txt)
