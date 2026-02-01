import json
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
SEARCH_URL = "https://www.ratemyprofessors.com/search/professors/825?q=*&did=11"
MAX_PROFESSORS = 209
HEADLESS = True
OUTPUT_FILE = 'rmp_rutgers_cs_data.json'

class RateMyProfessorScraper:
    def __init__(self, headless=True):
        options = webdriver.ChromeOptions()
        if headless: options.add_argument('--headless')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
        self.data = []

    def get_professor_list(self, url):
        print(f"\n Fetching list of professors...")
        self.driver.get(url)
        
        while True:
            current_cards = self.driver.find_elements(By.XPATH, "//a[contains(@class, 'TeacherCard__StyledTeacherCard')]")
            print(f"Professors found: {len(current_cards)}/{MAX_PROFESSORS}")
            
            if len(current_cards) >= MAX_PROFESSORS:
                break

            try:
                # The "Show More" button on the main search page
                show_more_btn = self.wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Show More')]")))
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", show_more_btn)
                time.sleep(1)
                self.driver.execute_script("arguments[0].click();", show_more_btn)
                time.sleep(2)
            except TimeoutException:
                break

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        cards = soup.find_all('a', class_=re.compile(r'TeacherCard__StyledTeacherCard'))
        
        professors = []
        for card in cards[:MAX_PROFESSORS]:
            prof_url = card.get('href')
            if prof_url:
                full_url = 'https://www.ratemyprofessors.com' + prof_url
                name_div = card.find('div', class_=re.compile(r'CardName__StyledCardName'))
                name = name_div.get_text(separator=" ", strip=True) if name_div else "Unknown"
                professors.append((name, full_url))
        return professors

    def expand_reviews_to_target(self, target_count):
        """Clicks 'Load More Ratings' and prints progress."""
        print(f"Goal: {target_count} reviews. Expanding...")
        last_count = 0
        
        while True:
            # re-count currently visible reviews
            current_cards = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'Rating__StyledRating')]")
            current_count = len(current_cards)
            
            # progress checker for inspection
            print(f"Progress: {current_count}/{target_count}")

            if current_count >= target_count or (current_count == last_count and last_count > 0):
                if current_count >= target_count: print("Target reached.")
                else: print("No more ratings to load.")
                break
            
            last_count = current_count

            try:
                # Clicking the "Load More Ratings" button on the profile page
                btn = self.wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Load More Ratings')]")))
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                time.sleep(1)
                self.driver.execute_script("arguments[0].click();", btn)
                time.sleep(2)
            except:
                break

    def scrape_professor_page(self, url):
        self.driver.get(url)
        time.sleep(2)
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        count_tab = soup.find('li', class_=re.compile(r'TeacherRatingTabs__StyledTab'))
        expected_count = int(re.search(r'(\d+)', count_tab.text).group(1)) if count_tab else 0
        
        self.expand_reviews_to_target(expected_count)
        
        # Final Extraction
        final_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # name extraction
        prof_name_tag = final_soup.find('h1', class_=re.compile(r'NameTitle__Name'))
        # space seperator for nammes
        prof_name = prof_name_tag.get_text(separator=" ", strip=True) if prof_name_tag else "Unknown"
        
        reviews = []
        rating_containers = final_soup.find_all('div', class_=re.compile(r'Rating__StyledRating'))
        
        for container in rating_containers:
            try:
                date_tag = container.find('div', class_=re.compile(r'TimeStamp__StyledTimeStamp'))
                date = date_tag.get_text(strip=True) if date_tag else "N/A"

                nums = container.find_all('div', class_=re.compile(r'CardNumRating__CardNumRatingNumber'))
                q = nums[0].get_text(strip=True) if len(nums) > 0 else "N/A"
                d = nums[1].get_text(strip=True) if len(nums) > 1 else "N/A"
                
                meta_items = container.find_all('div', class_=re.compile(r'MetaItem__StyledMetaItem'))
                meta_map = {"Grade": "N/A", "Attendance": "N/A", "Would Take Again": "N/A"}
                for item in meta_items:
                    parts = item.get_text(strip=True).split(":")
                    if len(parts) == 2:
                        label, val = parts[0].strip(), parts[1].strip()
                        if label in meta_map: meta_map[label] = val

                course = container.find('div', class_=re.compile(r'RatingHeader__StyledClass')).get_text(strip=True)
                comment = container.find('div', class_=re.compile(r'Comments__StyledComments')).get_text(strip=True)
                
                reviews.append({
                    "date": date,
                    "course": course,
                    "quality": q,
                    "difficulty": d,
                    "attendance": meta_map["Attendance"],
                    "grade": meta_map["Grade"],
                    "would_take_again": meta_map["Would Take Again"],
                    "comment": comment
                })
            except: continue

        return {"professor_name": prof_name, "reviews": reviews}

    def scrape_all(self):
        prof_list = self.get_professor_list(SEARCH_URL)
        for name, url in prof_list:
            print(f"\n--- Scraping Professor: {name} ---")
            self.data.append(self.scrape_professor_page(url))
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"\n Data successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    scraper = RateMyProfessorScraper(headless=HEADLESS)
    try:
        scraper.scrape_all()
    finally:
        scraper.driver.quit()