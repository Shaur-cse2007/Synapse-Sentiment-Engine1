import nltk
import os

# Define the directory where NLTK data will be stored
# This must be a location inside your project that Vercel can access.
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Set the NLTK data path environment variable for the script
# This is crucial so that Vercel can find the data later.
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

print(f"NLTK data path set to: {NLTK_DATA_PATH}")

# List the datasets your project needs
# Replace 'punkt' and 'stopwords' with the actual datasets your code uses.
datasets = ['punkt', 'stopwords', 'vader_lexicon']

for dataset in datasets:
    try:
        print(f"Attempting to download {dataset}...")
        nltk.download(dataset, download_dir=NLTK_DATA_PATH)
        print(f"{dataset} downloaded successfully.")
    except Exception as e:
        print(f"Error downloading {dataset}: {e}")
        # Exit the script with an error to fail the build if a critical dataset is missing
        exit(1)

print("NLTK data preparation complete.")
