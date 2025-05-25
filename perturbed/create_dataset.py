import gc
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
import openai
from openai import OpenAI
from llm_utils import LLM, generate_captions, generate_captions_async, FormatException
import time

import base64
import io
import cv2
import numpy as np
from PIL import Image
import re
from pprint import pprint
import json
import asyncio
import nest_asyncio
nest_asyncio.apply()

openai.api_key = os.getenv("OPENAI_API_KEY")
NUM_CAPTIONS = 10


prompt = """  
Imagine you have been assigned the task of progressively enhancing the following caption by systematically introducing unique and differentiating details:  

**Original Caption:**  
"{caption}"  

### **Task Overview:**  
You will generate **10 increasingly different variations** of this caption, ensuring that each version changes the semantic meaning of the **original caption**. If an image is provided, ensure that the changes to the caption are semanticaly different **distinct from the visual elements in the image**.  

### **Definition of Changing Semantic Meaning**  
Changing semantic meaning means that the modified caption should **alter the image if used in a generation model**.

This can be achieved by changing visual cues of the original caption including but not limited to:  
  - Identity of objects or people
  - Textures of objects or landscape elements
  - Location, time of day, weather, or environment specifics

### **Task Breakdown & Structure:**  
1. **Incremental Enhancement:**  
   - Generate **10 versions** of the caption.  
   - Each version should introduce an increasing amount of semantic differences by increments of **[10, 20, ..., 100] (in percentage)*.  

2. **Gradual Transformation:**  
   - Ensure that each step logically builds upon the previous one.  
   - The final version should have a completely different semantic meaning from the original caption.

3. **Handling Image Input (if provided):**  
   - If an image is provided, ensure that **the semantics of the changed captions are different from the visual elements in the image**.  

4. **Output Formatting:**  
   - Each caption should be **separated by a consistent delimiter** to ensure clarity.  
   - Use the following format for **each generated caption:**  
   - Caption N - Uniqueness Percentage%: Generated Caption  
   - Ensure that each step **logically evolves** from the previous version, creating a seamless and natural transformation.  

### **Expected Output Format Example:**  
*Input Caption*: Golden hues gently stretch across the horizon, deepening as the sun slowly dips, casting soft amber reflections on the tranquil sea. 

Caption 1 - 10%: Crimson and violet hues gently stretch across the horizon, deepening as the sun slowly dips, casting reflections on the tranquil sea. 

Caption 2 - 20%: Crimson and violet hues gently stretch across the horizon, deepening as the sun slowly dips, casting reflections on the waves. 

Caption 3 - 30%: Crimson and violet hues gently stretch across the horizon, deepening as the sun rises, casting reflections on the waves.
<Captions 4-10 omitted for brevity>

### **Goal:**  
By the end, the series of 10 captions should **illustrate a clear evolution** in semantic meaning both in terms of text and any provided image.  

---  
**Input Parameters:**  
- **Caption:** "{caption}"  
- **(Optional) Image:** A visual reference that must also be considered when introducing unique details.  

Your task is to ensure that each new version would generate an image that is **perceptibly different** from both the original caption and any provided visual input.  
"""


def extract_sentences(captions):
    captions_dict = {}
    
    for line in captions:
        split_line = line.split(":", 2)
        captions_dict[split_line[0]] = split_line[1].strip()
    
    return captions_dict


def get_image_data_uri(image):
    """
    Retrieves an image from a function, encodes it in Base64, 
    and returns the image data URI for use in APIs.

    Args:
        image_function (function): A function that returns an image (PIL or OpenCV format).

    Returns:
        str: The Base64-encoded image data URI.
    """
    def encode_image(image):
        """Encodes an OpenCV (NumPy array) or PIL image into Base64 format."""
        if isinstance(image, Image.Image):  # PIL image
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")  # Convert to JPEG format
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        elif isinstance(image, np.ndarray):  # OpenCV image
            _, buffer = cv2.imencode('.jpg', image)  # Convert to JPEG format
            return base64.b64encode(buffer).decode('utf-8')
        else:
            raise ValueError("Unsupported image type. Must be PIL or OpenCV format.")


    # Encode the image
    base64_image = encode_image(image)

    # Return the Base64 image as a data URI
    return f"data:image/jpeg;base64,{base64_image}"


def perturb_using_chatgpt(text, image=None):
    
    if image is not None:
        image_uri =  get_image_data_uri(image)
        api_messages_schema = [{
            "role": "user",
            "content": prompt.format(caption=text),
        }, {
            "role": "user",
            "content": image_uri,
        }]
    else:
        api_messages_schema = [{
            "role": "user",
            "content": prompt.format(caption=text),
        }]
    response = generate_captions(api_messages_schema)
    return response

async def perturb_using_chatgpt_async(all_text_image):
    tasks = []
    for text, image in all_text_image:
        if image is not None:
            image_uri =  get_image_data_uri(image)
            api_messages_schema = [{
                "role": "user",
                "content": prompt.format(caption=text),
            }, {
                "role": "user",
                "content": image_uri,
            }]
        else:
            api_messages_schema = [{
                "role": "user",
                "content": prompt.format(caption=text),
            }]
        tasks.append(generate_captions_async(api_messages_schema))
    results = await tqdm.gather(*tasks, desc=f"Getting data")
    return results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake",   action="store_true")
    parser.add_argument("--num_samples",    type=int, default=1024)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--pool",           type=str, default='avg', choices=['avg', 'cls'])
    parser.add_argument("--prompt",         action="store_true")
    parser.add_argument("--dataset",        type=str, default="minhuh/prh")
    parser.add_argument("--subset",         type=str, default="wit_1024")
    parser.add_argument("--caption_idx",    type=int, default=0)
    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--modality",       type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--output_dir",     type=str, default="./results/features")
    parser.add_argument("--num_workers",    type=int, default=10)
    parser.add_argument("--save_path",      type=str, default="perturbed/data.json")
    parser.add_argument("--qlora",          action="store_true")

    args = parser.parse_args()

    if args.qlora:
        print(f"QLoRA is set to True. The alignment score will be slightly off.")

    
    # load dataset once outside    
    dataset = load_dataset(args.dataset, revision=args.subset, split='train')
    print("dataset loaded")
    
    texts = [str(x['text'][args.caption_idx]) for x in dataset]
    # Load pre-trained Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    all_similarities = []
    for data_idx in range(0, len(dataset), args.num_workers):
        max_data_idx = min(data_idx + args.num_workers, len(dataset))
        indices = [idx for idx in range(data_idx, max_data_idx)]
        all_text_image = [(dataset[idx]['text'][args.caption_idx], None) for idx in indices]

        try:
            with open(args.save_path, "r") as file:
                data = json.load(file)  # Load existing data
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        start_time = time.time()
        for _ in range(5):
            try:
                batch_data = []
                all_responses = asyncio.run(perturb_using_chatgpt_async(all_text_image))
                end_time = time.time()
                print(f"Time taken: {end_time - start_time}")
                for idx, (text, _), response in zip(indices, all_text_image, all_responses):
                    captions = response.choices[0].message.parsed.unique_captions
                    if len(captions) != NUM_CAPTIONS:
                        raise FormatException
                    sentences = extract_sentences(captions)
                    
                    pprint(sentences)
                    orig_embedding = model.encode(text)
                    similarities = []
                    for _, sentence in sentences.items():
                        # Get the embeddings for each sentence
                        perturbed_embedding = model.encode(sentence)

                        # Compute cosine similarity between the embeddings
                        similarity = cosine_similarity([orig_embedding], [perturbed_embedding])
                        similarities.append(similarity.item())

                    sentences["Caption 0 - 0%"] = text
                    batch_data.append({f"Index - {idx}": {"sentences": sentences, "similarities": similarities}})
                    all_similarities.append(similarities)
                
                # Successfully terminated
                data.extend(batch_data)
                break
            except FormatException as e: 
                print(e)
        
        with open(args.save_path, "w") as file:
            json.dump(data, file, indent=4)
        print("New nested entry added successfully!")
    np_similarities = np.array(all_similarities)
    print(np_similarities.mean(axis=0))
    
    


        
    
    
    
    
