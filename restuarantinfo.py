import pandas as pd
import requests
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import os

API_KEY = os.environ.get("GOOGLE_API_KEY")  # Replace with your actual API key
OUTPUT_FILE = "categorized_restaurants_with_details.csv"  # Output CSV file

FIND_PLACE_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# Create a session for performance and reusability
session = requests.Session()

def find_place_id(restaurant_name, api_key):
    params = {
        "input": restaurant_name,
        "inputtype": "textquery",
        "fields": "place_id",
        "key": api_key,
        "locationbias": "circle:20000@37.7749,-122.4194"
    }
    try:
        response = session.get(FIND_PLACE_URL, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json().get("candidates", [])
            if result:
                return result[0].get("place_id")
    except requests.exceptions.RequestException as e:
        print(f"Error finding place ID for {restaurant_name}: {e}")
    return None

def get_place_details(place_id, api_key):
    """Get detailed information for a given Place ID."""
    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,price_level,types,website,formatted_phone_number,rating,user_ratings_total,opening_hours",
        "key": api_key
    }
    try:
        response = session.get(PLACE_DETAILS_URL, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("result", {})
    except requests.exceptions.RequestException as e:
        print(f"Error getting details for place ID {place_id}: {e}")
    return {}

def classify_service_type(types):
    """Classify the service type based on 'types'."""
    types = [t.lower() for t in types]
    if "restaurant" in types and "bar" not in types and "fast_food" not in types:
        return "BB1"  # Full service restaurant
    elif "fast_food" in types or "meal_takeaway" in types or "meal_delivery" in types:
        return "BB2"  # Quick service restaurant
    elif "bar" in types:
        return "BB3"  # Bar
    elif any(keyword in types for keyword in ["cafe", "bakery", "food", "drink"]):
        return "BB2" if "restaurant" not in types else "BB1"
    else:
        return "Not a restaurant"

def clean_name(restaurant_name):
    # Instead of removing 'greater NYC area', explicitly add 'in San Francisco'
    cleaned = restaurant_name.replace("greater NYC area", "").strip()
    return f"{cleaned}"

def extract_emails(text):
    """Extract and return email addresses from text."""
    pattern = r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b'
    return set(re.findall(pattern, text))

def detect_reservation_platform(html_content):
    """Detect the reservation platform from the given HTML content."""
    if 'id="resy_button_container"' in html_content or "widgets.resy.com" in html_content or "resy.com" in html_content or "Resy" in html_content:
        return "Resy"
    if "OpenTable" in html_content or "opentable.com" in html_content:
        return "OpenTable"
    if "Tock" in html_content or "exploretock.com" in html_content:
        return "Tock"
    return ""

def scrape_emails_and_pos_from_website(start_url, max_links=10):
    """Scrape emails, POS system, loyalty programs, and reservation platform from a given website and up to 10 pages on the same domain."""
    emails_found = set()
    pos_system = ""
    loyalty_programs = []
    reservation_platform = ""

    to_scrape = [start_url]
    visited = set()
    base_domain = urlparse(start_url).netloc
    links_scraped = 0

    while to_scrape and links_scraped < max_links:
        url = to_scrape.pop(0)
        if url in visited:
            continue
        visited.add(url)
        
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            html_content = soup.prettify()

            # Extract emails
            new_emails = extract_emails(text)
            cleaned_emails = set()
            for email in new_emails:
                if '@' in email and '.com' in email:
                    # Cut off everything after '.com'
                    idx = email.find('.com')
                    # Include the '.com' in final email
                    cleaned_email = email[:idx+4]
                    cleaned_emails.add(cleaned_email)
                else:
                    cleaned_emails.add(email)
            
            emails_found.update(cleaned_emails)

            # Check for POS systems and loyalty programs
            if "www.toasttab.com" in html_content and pos_system != "Toast":
                pos_system = "Toast"
            if "inkindscript.com" in html_content and "inKind" not in loyalty_programs:
                loyalty_programs.append("inKind")
            if "spoton.com" in html_content and "SpotOn" not in loyalty_programs:
                loyalty_programs.append("SpotOn")

            # Detect reservation platform more thoroughly
            if not reservation_platform:
                platform = detect_reservation_platform(html_content)
                if platform:
                    reservation_platform = platform

            # Gather links from this page
            priority_links = []
            normal_links = []
            for link in soup.find_all('a', href=True):
                absolute_link = urljoin(url, link['href'])
                parsed_link = urlparse(absolute_link)

                # Only follow links within the same domain
                if parsed_link.netloc == base_domain:
                    link_text = link.get_text().lower()
                    link_url = absolute_link.lower()
                    if any(keyword in link_text or keyword in link_url for keyword in ["reservation", "book", "resy", "opentable", "tock"]):
                        priority_links.append(absolute_link)
                    else:
                        normal_links.append(absolute_link)

            # Add priority links first
            for pl in priority_links:
                if len(visited) + len(to_scrape) < max_links + 1:
                    to_scrape.append(pl)

            # Then add normal links
            for nl in normal_links:
                if len(visited) + len(to_scrape) < max_links + 1:
                    to_scrape.append(nl)

            links_scraped += 1

        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")
            continue

    # Remove duplicates from loyalty_programs
    loyalty_programs = list(set(loyalty_programs))

    return emails_found, pos_system, '; '.join(loyalty_programs), reservation_platform

def main(input_file):
    # Load the CSV file
    print(f"Loading input file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Number of rows in input file: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    if "whole name" not in df.columns:
        print("Error: 'whole name' column not found in the input CSV.")
        return

    if df.empty:
        print("The input CSV is empty. No restaurants to process.")
        return

    # Prepare output data
    detailed_data = []
    
    # Try loading partial progress if exists
    try:
        existing_df = pd.read_csv(OUTPUT_FILE)
        processed_names = set(existing_df["whole name"].unique())
        detailed_data = existing_df.to_dict('records')
        print(f"Loaded existing progress with {len(existing_df)} records.")
    except FileNotFoundError:
        processed_names = set()

    # Process each restaurant
    for index, row in df.iterrows():
        restaurant_name = clean_name(row["whole name"])

        if restaurant_name in processed_names:
            # Already processed
            continue

        print(f"Processing: {restaurant_name}")

        # Step 1: Find Place ID
        place_id = find_place_id(restaurant_name, API_KEY)
        if not place_id:
            print(f"Place ID not found for: {restaurant_name}")
            pd.DataFrame(detailed_data).to_csv(OUTPUT_FILE, index=False)
            continue

        print(f"Found place ID for {restaurant_name}: {place_id}")

        # Step 2: Get Place Details
        details = get_place_details(place_id, API_KEY)
        if not details:
            print(f"Details not found for Place ID: {place_id} ({restaurant_name})")
            pd.DataFrame(detailed_data).to_csv(OUTPUT_FILE, index=False)
            continue

        print(f"Got details for {restaurant_name}: {details.get('name')}")

        # Step 3: Classify Service Type
        service_type = classify_service_type(details.get("types", []))

        # Extract website
        website = details.get("website")
        if website:
            print(f"Scraping website for {restaurant_name}: {website}")
        else:
            print(f"No website found for {restaurant_name}.")

        # Scrape emails if a website is available
        if website:
            emails, pos_system, loyalty_programs, reservation_platform = scrape_emails_and_pos_from_website(website, max_links=10)
        else:
            emails, pos_system, loyalty_programs, reservation_platform = set(), "", "", ""

        # If no emails found, create an empty entry
        if not emails:
            emails = [""]

        for email in emails:
            entry = {
                "whole name": row["whole name"],
                "Name": details.get("name"),
                "Address": details.get("formatted_address"),
                "Price Level": details.get("price_level"),
                "Types": ", ".join(details.get("types", [])),
                "Category": service_type,
                "Website": website,
                "Phone Number": details.get("formatted_phone_number"),
                "Rating": details.get("rating"),
                "Review Count": details.get("user_ratings_total"),
                "Opening Hours": ", ".join(details.get("opening_hours", {}).get("weekday_text", [])),
                "Email": email,
                "POS System": pos_system,
                "Loyalty Programs": loyalty_programs,
                "Reservation Platform": reservation_platform
            }
            detailed_data.append(entry)

        # Update processed_names and save partial progress
        processed_names.add(restaurant_name)
        pd.DataFrame(detailed_data).to_csv(OUTPUT_FILE, index=False)
        print(f"Progress saved after processing {restaurant_name}.")

    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    input_file = "input.csv"  # Replace 'input.csv' with your actual input file name
    main(input_file)
