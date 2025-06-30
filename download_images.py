import re
from icrawler.builtin import BingImageCrawler

# Function to sanitize folder names for Windows
def sanitize_folder_name(name):
    # Replace invalid characters with underscores
    return re.sub(r'[\\/*?:"<>|]', "_", name)

# List of fruit diseases
diseases = [
    "Apple : Black Rot",
    "Apple : Cedar rust",
    "Apple : Scab",
    "Apple : Healthy",
    "Apple : Bitter Rot",]

#for disease in diseases:
for disease in diseases:
    folder_name = sanitize_folder_name(disease.replace(" ", "_"))
    crawler = BingImageCrawler(storage={'root_dir': f'grape_dataset/{folder_name}'})
    crawler.crawl(
        keyword=disease, 
        max_num=200,  # Increase this number
        min_size=(200, 200)  # Avoid small/low-quality images
    )


