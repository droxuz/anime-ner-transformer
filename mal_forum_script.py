import time
from selenium import webdriver
from selenium.webdriver.common.by import By
MAL = "https://myanimelist.net/forum/?board=16&show="
iterator = 0
driver = webdriver.Chrome()
driver.get(MAL+str(iterator))
wrapper = driver.find_element(By.TAG_NAME, 'tr')
posting = wrapper.find_elements(By.CLASS_NAME, 'forum_boardrow1')
navigable_link = posting.find_elements(By.TAG_NAME, 'a')





time.sleep(2)

driver.quit()