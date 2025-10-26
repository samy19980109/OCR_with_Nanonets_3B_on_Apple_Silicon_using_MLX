from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

def resize_image(input_path, output_path, size=(1024, 1024)):
     img = Image.open(input_path)
     img = img.convert("RGB")
     img.thumbnail(size)
     img.save(output_path)
     return output_path


model_name = "mlx-community/Nanonets-OCR2-3B-4bit"
model, processor = load(model_name)
config = load_config(model_name)

# Define the prompt
prompt = """
Extract all clearly printed text and structured fields from this trading card image. Focus on the following details:
	•	Card Title (e.g., player name)
	•	Player’s Team
	•	Card Series/Release (e.g., 2021 Topps Fire)
	•	Card Subset (e.g., Scorching Signatures)
	•	PSA or grading company label and grade (e.g., PSA 9 MINT)
	•	Card number/serial (e.g., #SSVGJ or other identifier)
	•	Certification or serial number (e.g., 122390435)
	•	Any autograph details (if present, confirm it says ‘Certified Autograph Issue’)
	•	Any visible text overlay or background text

Return results as a structured JSON object with fields: {“Player Name”: “”,“Team”: “”,“Card Series”: “”,“Card Subset”: “”,“Grading Company”: “”,“Grade”: “”,“Card Number”: “”,“Certification Number”: “”,“Autograph Details”: “”,“Overlay Text”: “”} If a field is not visible, set its value to ‘Not visible’.”
"""

# TODO: Change this to your own image path
image_paths = ["/Users/samarthagarwal/dev/MLX/Goldin_images/Original/Vladdy/Front.jpg"]

# Dynamically Construct paths for resized images
resized_image_paths = [
    image_path.replace("/Goldin_images/Original/", "/Goldin_images/Resized/")
    for image_path in image_paths
]

# Resize the image
for image_path, resized_image_path in zip(image_paths, resized_image_paths):
    resize_image(image_path, resized_image_path)

formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image_paths))

result = generate(model, processor, formatted_prompt, resized_image_paths, max_tokens=128, verbose=True)
# print(result)
