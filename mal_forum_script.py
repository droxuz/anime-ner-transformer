import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import json
# Traverses the recommendations forum of the MAL website
# Collects all the posts from pages selected and makes a list of all links to posts
finetune_path = "data/anime_training_data/finetune_prompt.jsonl"
MAL = "https://myanimelist.net/forum/?board=16&show="
link_list = []
iterator = 0
max_pages = 150
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)
for iterator in range(0, max_pages, 50):
    driver.get(MAL+str(iterator))
    table_rows = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tr[id^='topicRow']")))
    for row in table_rows:
        postings = row.find_element(By.CSS_SELECTOR, "td.forum_boardrow1")
        navigable_link = postings.find_element(By.TAG_NAME, 'a')
        href = navigable_link.get_attribute("href")
        link_list.append(href)
    time.sleep(5)


# Traverse the links to extract each text content of the post from the recommendations 

recommendation_list = []
for link in link_list[1:]:
    driver.get(link)
    try:
        posting_text = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table[id^='message'] > tbody > tr > td")))
        text = posting_text.text.strip()
        if text:
            recommendation_list.append(text)
    except TimeoutException:
        print(f"Timed out")
    time.sleep(10)
driver.quit()

def create_jsonl(recommendation_list, path):
    with open(path, "w", encoding= "UTF-8")as file:
        for line in recommendation_list:
            file.write(json.dumps(line, ensure_ascii= False, allow_nan= False)+ "\n")

create_jsonl(recommendation_list, finetune_path)

