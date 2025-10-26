from PIL import Image

img = Image.open("/Users/samarthagarwal/dev/MLX/Goldin_images/Vladdy/Front.jpg")
img = img.convert("RGB")
img.thumbnail((1024, 1024))
img.save("/Users/samarthagarwal/dev/MLX/Goldin_images/Resized/Vladdy/Front.jpg")