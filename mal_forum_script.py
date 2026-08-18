import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Traverses the recommendations forum of the MAL website
# Collects all the posts from pages selected and makes a list of all links to posts
MAL = "https://myanimelist.net/forum/?board=16&show="
link_list = []
iterator = 0
max_pages = 50
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
    post_table = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table[id^='message']")))
    for post in post_table:
        text = post.find_element(By.CSS_SELECTOR, ':scope > tbody > tr > td')
        post_text = text.text.strip()
        recommendation_list.append(post_text)
    time.sleep(10)
print(recommendation_list[0])
driver.quit()
