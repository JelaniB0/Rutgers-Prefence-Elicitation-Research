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
        '__zlcmid': '1UYo3PTsRKJ3eFE',
        '_clck': 'wavkdz%7C2%7Cfvf%7C0%7C1856',
        '_ga': 'GA1.2.2023116982.1737663962',
        '_ga_29NEBSX93J': 'GS1.1.1743740933.3.0.1743740938.0.0.0',
        '_ga_2J08ESZEHV': 'GS1.1.1737692850.1.0.1737692855.0.0.0',
        '_ga_2L8851K40M': 'GS1.1.1745091583.2.0.1745091583.0.0.0',
        '_ga_376294733': 'GS1.1.1745294951.4.1.1745295273.0.0.0',
        '_ga_3K77D0NWXZ': 'GS1.1.1743599759.1.0.1743599761.0.0.0',
        '_ga_5EBF2S2KRB': 'GS1.1.1745429516.2.0.1745429520.0.0.0',
        '_ga_6HQVYQ38G0': 'GS1.2.1745771709.3.0.1745771709.0.0.0',
        '_ga_B6BCKQE1XW': 'GS1.1.1743621331.2.1.1743621373.0.0.0',
        '_ga_DVCN4VQT8D': 'GS1.1.1745130296.18.0.1745130304.0.0.0',
        '_ga_F0X7Y46YTM': 'GS1.1.1740017379.2.0.1740017379.0.0.0',
        '_ga_F78K85ZPMG': 'GS1.1.1742623016.2.1.1742623134.0.0.0',
        '_ga_FJJ06ZV9LX': 'GS1.1.1742066760.1.0.1742066761.0.0.0',
        '_ga_H4R35LHY9K': 'GS1.1.1743913829.6.0.1743913829.0.0.0',
        '_ga_J1DSRJRQZ1': 'GS1.1.1745294953.5.1.1745295274.0.0.0',
        '_ga_JCHSMVR2MB': 'GS1.2.1742066424.1.1.1742066748.0.0.0',
        '_ga_K7N8GCP9CJ': 'GS1.1.1743620824.3.0.1743620826.58.0.0',
        '_ga_KMRD2XMMJR': 'GS1.1.1738893639.4.0.1738893667.0.0.0',
        '_ga_M6FS8HG1PG': 'GS1.1.1745698308.9.1.1745698458.0.0.0',
        '_ga_NQ7HLJ30E3': 'GS1.1.1745429516.2.0.1745429520.0.0.0',
        '_ga_Q9231LEHXD': 'GS1.1.1745294495.5.1.1745295275.0.0.0',
        '_ga_QNY9B4GYMW': 'GS1.1.1745120430.11.1.1745120466.0.0.0',
        '_ga_RX0G5XBWJW': 'GS1.1.1743599758.1.0.1743599761.0.0.0',
        '_ga_V859C9HTED': 'GS1.1.1745089289.1.1.1745089307.0.0.0',
        '_ga_XYDNZKGXH3': 'GS1.1.1745185795.1.1.1745185809.0.0.0',
        '_ga_YD1YBHNY8G': 'GS1.1.1744950829.1.1.1744950891.0.0.0',
        '_ga_YM2HRZKGGN': 'GS1.1.1740441558.1.1.1740441623.0.0.0',
        '_ga_ZYY85J4EN8': 'GS1.1.1740441558.2.1.1740441614.0.0.0',
        '_gid': 'GA1.2.296121471.1769280995',
        '_hjSessionUser_5064936': 'eyJpZCI6IjEyNDZmODY0LWZiOTItNTgyZS1hZjA4LWM5OTY4MTQ2MjFlZSIsImNyZWF0ZWQiOjE3NDAxMTkwNjUzNjUsImV4aXN0aW5nIjp0cnVlfQ==',
        'AMCV_8E929CC25A1FB2B30A495C97%40AdobeOrg': '179643557%7CMCIDTS%7C20375%7CMCMID%7C53783292193496976344189685377209500578%7CMCAID%7CNONE%7CMCOPTOUT-1760329670s%7CNONE%7CvVersion%7C5.5.0',
        'BCSessionID': 'aa47432f-416f-4b8d-9d7e-dd7c6cc51cce',
        'EssUserTrk': '43850750.6496b764e38be',
        'fpestid': 'XdGidS05m50ugXIhyJJ-S0vy4mWRmzHjtZgE4FMrTFOXTQhOLRf44IruOAD9-F9kYfsuAg',
        'kndctr_8E929CC25A1FB2B30A495C97_AdobeOrg_identity': 'CiY1Mzc4MzI5MjE5MzQ5Njk3NjM0NDE4OTY4NTM3NzIwOTUwMDU3OFIQCLqS-dqdMxgBKgNWQTYwA_ABupL52p0z',
        'twk_uuid_5ae85beb227d3d7edc24dcb1': '%7B%22uuid%22%3A%221.70jA5nQC2E3aigSKFQO830uxXsQ1PzyFA7JgRA2w72qnvmW1gG59OAE0is1EB1GW0d21c7Xb0y7g1UB8uuPU0LhUf2RXtCUTTxuV8NFtx3KVFQdAtq5l%22%2C%22version%22%3A3%2C%22domain%22%3A%22rutgers.edu%22%2C%22ts%22%3A1769030835424%7D',
        'utag_main': 'v_id:0199db5def6b002008b1c36238780506f006106700bd0$_sn:1$_se:51$_ss:0$_st:1760324261421$ses_id:1760321990508%3Bexp-session$_pn:28%3Bexp-session$vapi_domain:rutgers.edu',
        '.DeAcFramework': '4DEF81F3159935206F7A4505BDFA933E20E22DB9224BAE8D8AAD8A1CF9617A41FC08321D3C8A6DD78323E2777014F877EE128559888DA463AA1D88EE5BB099CA5255182F2B24E6A46E633F9B008F31E1B64E8E609893497FC1008C06AC9F1FA1934A46F38823D31B3B8F348D64F9F918C14A8AF17896C038BABA16F28AD0ABC1',
        'ASP.NET_SessionId': 'mdayv1dwo1k1ho3ezmvftp5o',
        'BIGipServerdn_os_ver_12-http-Pool': '720313772.20480.0000',
    }
    
    if not cookies or all(not v for v in cookies.values()):
        print("\nNo cookies provided!")
    else:
        print("Fetching courses...")
        courses, soup, session, headers = scrape_rutgers_courses(cookies)
        
        if courses:
            print(f"\n✓ Found {len(courses)} courses!")
            
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