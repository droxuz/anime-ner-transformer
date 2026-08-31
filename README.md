# Anime MLM 
### Creating a NAMED ENTITY RECOGNITION to assist with finding anime recommendations

This repository outlines the process of creating an anime recommendation system from scratch, utilizing different methods of recommendation via content based filtering with TF-IDF and cosine similarity, transformer based NLP to create a NER to find fields of which can be plugged into a content based filtering system. I watched anime on the regular and saw no recommendation systems that recommend based on plot points, and themes such Made in Abyss, or Heavenly Delusion. These 2 shows have really unique world building environments and themes. Thus this project was made to find that perfect match of animes, I know that using a strict content based filtering system is the better call by letting users link their MAL accounts to form a cosine vector and compare multiple genres, ratings, and other fields but I wanted to experiment with NLP and Neural Networks. Leading to a somewhat 50/50 named entity recognition.

## Specifications
Using a Kaggle MAL anime dataset (No longer available) used the synopses to train a 6 layer encoder only transformer model for the purpose to create a masked language model similar to BERT for the specific use of finetuning on a smaller set of data of recommendation prompts from MAL, with the end goal of finding BIO tags of TITLE, CONSTRAINT, ATTRIBUTE, FORMAT.

Utilizing the MAL dataset, synopses were truncated to 1000 characters and had to be cleaned and processed for it to be usable such as cleaning out [SOURCE: MyAnimeList], and unfinished synopses due to the truncation. With the use of Regex, Dataframes, and JSON to create a jsonl file.

From the cleaned synopses the 80/20 training/validation split the training data was then used to train a BPE tokenizer and subsequently tokenized the training data. From both sets of data to create dataset that involves attention mask, padding mask, and input ids.

In training the model using plugging in the dataset into the model seeing the total losses of both training and validation and averaging them out for data visualization utilizing AdamW optimizer to optimize training weights. The end result is a really lossy and overfitted but usable masked language model for finetuning.

<img width="560" height="400" alt="image" src="https://github.com/user-attachments/assets/e5d76ba8-9c00-4e49-8327-0900ea85876e" />  

_Image of Masked Language Model Losses after Training_

For the uses of finetuning the main idea is finding really moderate size and good quality examples and training utilizing either LoRa or full weight adjustments, considering the smaller model size full weight adjustments were used and somewhat good outcomes considering the CONSTRAINT, ATTRIBUTE, and FORMAT.

<img width="560" height="400" alt="image" src="https://github.com/user-attachments/assets/014ea757-6161-4e4a-ba3a-e63927273c10" />  

_Image of Fine Tuning Losses after Finetune_

## Results
After creating this little masked language model it taught me things about tokenization and the importance of cleaning data for training, but overall using a production model is more practical and is faster and easier to use. **Next steps either utilize a production model or change the structure for recommendations.** 

## Tech Stack
- **[Pytorch](https://pytorch.org/)** Model Architecture, Optimization, Dataset, Dataloader
- **[NumPy](https://numpy.org/doc/stable/#)**  Data Processing, General Utilities
- **[pandas](https://pandas.pydata.org/)**  Dataframes, Data Cleaning
- **[Jupyter](https://jupyter.org/)**  Experimentation, Analysis
- **[HuggingFace](https://huggingface.co/)**  Tokenizer
- **[Matplotlib](https://matplotlib.org/)**  Data Visualization
- **[Selenium](https://www.selenium.dev/)**  Web Driver
