import requests
from bs4 import BeautifulSoup
import json

def scrape_rutgers_courses(cookies_dict):
    """
    Scrape course information from Rutgers Degree Navigator
    
    Args:
        cookies_dict: Dictionary of cookies from your browser session
    
    Returns:
        List of course dictionaries
    """
    
    url = "https://dn.rutgers.edu/DN/Audit/ViewCategoryItems.aspx?pageid=DNViewCategoryItems&catid=750&degreeid=731"
    
    # Create a session with your cookies
    session = requests.Session()
    session.cookies.update(cookies_dict)
    
    # Add headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = session.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML using Python's built-in parser (no lxml needed)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        courses = []
        
        # Find the main results table
        results_table = soup.find('table', class_='DeAcGridView')
        
        if results_table:
            # Find all rows except the header
            rows = results_table.find('tbody').find_all('tr')
            
            print(f"Found {len(rows)} course rows")
            
            for row in rows:
                # Get all table cells
                cells = row.find_all('td')
                
                if len(cells) >= 2:  # Make sure we have at least code and title
                    course_data = {}
                    
                    # First cell is the course code
                    course_data['code'] = cells[0].get_text(strip=True)
                    
                    # Second cell is the course title
                    course_data['title'] = cells[1].get_text(strip=True)
                    
                    # Try to extract the item ID from the View link for more details
                    view_link = row.find('a', href=lambda x: x and 'ViewItem.aspx' in x)
                    if view_link:
                        href = view_link['href']
                        # Extract itemID from URL like "ViewItem.aspx?pageid=DNViewItem&itemID=49543&catid=750&degreeid=731"
                        import re
                        match = re.search(r'itemID=(\d+)', href)
                        if match:
                            course_data['item_id'] = match.group(1)
                            course_data['detail_url'] = f"https://dn.rutgers.edu/DN/Audit/{href}"
                    
                    courses.append(course_data)
        
        return courses, soup, session, headers
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
        return [], None, None, None


def fetch_course_details(session, detail_url, headers):
    """
    Fetch detailed information for a single course
    
    Args:
        session: Requests session with cookies
        detail_url: URL to the course detail page
        headers: Request headers
    
    Returns:
        Dictionary with description and credits
    """
    try:
        response = session.get(detail_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        details = {}
        
        # Look for the description and credits
        # The page typically has text like "FullDescription:..." and "Credits:..."
        page_text = soup.get_text()
        
        # Extract description
        import re
        desc_match = re.search(r'FullDescription:\s*(.+?)(?=Credits:|$)', page_text, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()
            # Clean up newlines and extra whitespace
            description = description.replace('\n', ' ').replace('\r', ' ')
            # Replace multiple spaces with single space
            description = re.sub(r'\s+', ' ', description)
            details['description'] = description.strip()
        
        # Extract credits
        credits_match = re.search(r'Credits:\s*(\d+(?:\.\d+)?)', page_text)
        if credits_match:
            details['credits'] = credits_match.group(1)
        
        return details
    
    except Exception as e:
        print(f"Error fetching details from {detail_url}: {e}")
        return {}    

if __name__ == "__main__":
    print("=" * 60)
    print("Rutgers Course Scraper")
    print("=" * 60)
    
    # cookies pasted here
    cookies = {
        
    }
    
    if not cookies or all(not v for v in cookies.values()):
        print("\nNo cookies provided!")
    else:
        print("Fetching courses...")
        courses, soup, session, headers = scrape_rutgers_courses(cookies)
        
        if courses:
            print(f"\n Found {len(courses)} courses!")
            
            # Now fetch details for each course
            print("\nFetching detailed information for each course...")
            for i, course in enumerate(courses, 1):
                if 'detail_url' in course:
                    print(f"  [{i}/{len(courses)}] Fetching details for {course['code']}...")
                    details = fetch_course_details(session, course['detail_url'], headers)
                    course.update(details)
                    
                    # Small delay to avoid overwhelming the server
                    import time
                    time.sleep(0.5)
            
            print("\nFirst few courses with details:")
            for i, course in enumerate(courses[:3], 1):
                print(f"\n{i}. {course['code']} - {course['title']}")
                if 'credits' in course:
                    print(f"   Credits: {course['credits']}")
                if 'description' in course:
                    desc_preview = course['description'][:100] + "..." if len(course['description']) > 100 else course['description']
                    print(f"   Description: {desc_preview}")
            
            # Save to JSON file
            with open('rutgers_courses.json', 'w') as f:
                json.dump(courses, f, indent=2)
            print(f"\n All courses with details saved to 'rutgers_courses.json'")
        else:
            print("\n No courses found. The page structure might be different.")
            if soup:
                print("\nSaving raw HTML for inspection...")
                with open('page_source.html', 'w', encoding='utf-8') as f:
                    f.write(str(soup.prettify()))
                print(" HTML saved to 'page_source.html' - inspect it to find the correct selectors")