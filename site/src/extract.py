from pathlib import Path
from typing import List
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PIL.Image import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

import os
import json
import time

def pdf_to_image(pdf_path):
	images = convert_from_path(pdf_path)
	return images

def tf_id_detection(image, model, processor):
	prompt = "<OD>"
	inputs = processor(text=prompt, images=image, return_tensors="pt")
	generated_ids = model.generate(
		input_ids=inputs["input_ids"],
		pixel_values=inputs["pixel_values"],
		max_new_tokens=1024,
		do_sample=False,
		num_beams=3
	)
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
	annotation = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
	return annotation["<OD>"]

def save_image_from_bbox(image, annotation, page, output_dir):
	# the name should be page + label + index
	for i in range(len(annotation['bboxes'])):
		bbox = annotation['bboxes'][i]
		label = annotation['labels'][i]
		x1, y1, x2, y2 = bbox
		cropped_image = image.crop((x1, y1, x2, y2))
		cropped_image.save(os.path.join(output_dir, f"{page}_{label}_{i}.png"))

def pdf_to_table_figures(page_images: List[Image], model_id: str = "yifeihu/TF-ID-large-no-caption", output_dir: str | Path = "./sample_output"):

	timestr = time.strftime("%Y%m%d-%H%M%S")
	output_dir = Path(output_dir) / f"output_{timestr}"
	output_dir.mkdir(parents=True, exist_ok=True)

	#TODO: Add logging.
	# print(f"PDF loaded. Number of pages: {len(page_images)}")
	model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
	processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

	for i, image in enumerate(page_images):
		annotation = tf_id_detection(image, model, processor)
		save_image_from_bbox(image, annotation, i, output_dir)
	



