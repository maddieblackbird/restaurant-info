import os
import re
import time
import requests
import pandas as pd
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Anthropic / Claude
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from anthropic import InternalServerError

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
API_KEY = os.environ.get("MAPS_API_KEY")                 # Google Maps API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # Claude API key (if available)

INPUT_FILE = "input.csv"                     # CSV with column "whole name"
OUTPUT_FILE = "categorized_restaurants_with_details.csv" # Output CSV file

# Base Google Places endpoints
TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# Create a session for performance and reusability
session = requests.Session()

# -------------------------------------------------------------------
# AI HELPER FUNCTIONS
# -------------------------------------------------------------------
def find_popular_dish(restaurant_name, reviews):
    """
    Use Anthropic to identify a popular dish/drink mentioned in the reviews.
    Returns a single dish/drink string.
    If ANTHROPIC_API_KEY is missing, returns a placeholder.
    """
    if not ANTHROPIC_API_KEY:
        return "[Missing Anthropic Key]"

    # Combine up to 5 reviews for context.
    relevant_reviews = "\n\n".join(r.get("text", "") for r in reviews[:5]).strip()

    # Prompt: Ask only for the top-mentioned dish or drink
    prompt = (
        f"{HUMAN_PROMPT}\n"
        "Identify a single dish or drink that seems most popular or most-mentioned across the following reviews. "
        "Return ONLY the name of that dish/drink (one line) with no extra words or commentary.\n\n"
        f"Restaurant name: {restaurant_name}\n\n"
        "Reviews:\n"
        f"{relevant_reviews}\n\n"
        f"{AI_PROMPT}"
    )

    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = anthropic_client.completions.create(
            prompt=prompt,
            max_tokens_to_sample=300,
            model="claude-2"
        )
        completion = response.completion.strip()
        return completion
    except InternalServerError as e:
        print(f"[DEBUG] Error calling Claude: {e}")
        return "[AI Error]"

def generate_intro(restaurant_name, reviews, popular_dish):
    """
    Generate a short personal-sounding email intro referencing the popular dish/drink from the reviews.
    Must describe an experience with friends (no romantic partners or anniversaries).
    Returns the intro_blurb or a placeholder if no Anthropic key.
    """
    if not ANTHROPIC_API_KEY:
        return "[Missing Anthropic Key for Intro]"

    # Combine up to 5 reviews for context.
    relevant_reviews = "\n\n".join(r.get("text", "") for r in reviews[:5]).strip()

    prompt = (
        f"{HUMAN_PROMPT}\n"
        "You are given the following details:\n"
        f"- Bar/Restaurant name: {restaurant_name}\n"
        f"- A popular dish/drink: {popular_dish}\n"
        " - Up to 5 recent reviews describing the vibe, ambiance, and service:\n"
        f"{relevant_reviews}\n\n"
        "Task:\n"
        "Compose a short, personal-sounding email intro (one or two sentences) that describes an experience "
        "you had with your friends. Mention that you ordered the above dish/drink and highlight details such as "
        "a cozy, welcoming ambiance or excellent service. DO NOT mention any significant others (like a spouse) "
        "or romantic events. Ensure the final text is not wrapped in quotation marks.\n\n"
        "Return only the email intro text with no additional commentary.\n"
        f"{AI_PROMPT}"
    )

    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = anthropic_client.completions.create(
            prompt=prompt,
            max_tokens_to_sample=300,
            model="claude-2"
        )
        completion = response.completion.strip()
        # Remove any extraneous quotation marks at the start and end
        if completion.startswith('"') and completion.endswith('"'):
            completion = completion[1:-1].strip()
        return completion
    except InternalServerError as e:
        print(f"[DEBUG] Error calling Claude for intro: {e}")
        return "[AI Error]"

# -------------------------------------------------------------------
# GOOGLE PLACES SEARCH FOR A SINGLE RESTAURANT IN NYC
# -------------------------------------------------------------------
def search_restaurant_in_nyc(restaurant_name, api_key):
    """
    Use Google's Text Search to find a specific restaurant by name,
    applying a location bias in New York City.
    
    Returns the most relevant result (dict) if found, otherwise None.
    """
    print(f"[DEBUG] Searching for '{restaurant_name}' in NYC using Text Search...")
    
    # You can tune location/radius for better relevance
    params = {
        "query": restaurant_name,
        "location": "40.7128,-74.0060",  # NYC center
        "radius": 30000,                # 30 km around NYC center
        "key": api_key
    }

    response = session.get(TEXT_SEARCH_URL, params=params, timeout=10)
    response_json = response.json()
    print(response_json)
    if response.status_code != 200:
        print(f"[DEBUG] Text Search returned status code {response.status_code}.")
        return None

    data = response.json()
    results = data.get("results", [])
    if not results:
        print("[DEBUG] No results found for this restaurant.")
        return None

    # Return the first result as the "best" match
    return results[0]

# -------------------------------------------------------------------
# PLACE DETAILS
# -------------------------------------------------------------------
def get_place_details(place_id, api_key):
    """
    Fetch a place's details (reviews, phone, website, etc.) from Google Places Details API.
    """
    print(f"[DEBUG] Getting details for Place ID: '{place_id}'")

    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,price_level,types,website,formatted_phone_number,"
                  "rating,user_ratings_total,opening_hours,reviews",
        "key": api_key
    }

    try:
        response = session.get(PLACE_DETAILS_URL, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json().get("result", {})
            # Find "most_relevant_review" by highest rating
            reviews = result.get("reviews", [])
            if reviews:
                top_review = sorted(reviews, key=lambda x: x.get("rating", 0), reverse=True)[0]
                result["most_relevant_review"] = {
                    "author_name": top_review.get("author_name"),
                    "text": top_review.get("text"),
                    "rating": top_review.get("rating"),
                }
            else:
                result["most_relevant_review"] = None
            return result
        else:
            print(f"[DEBUG] Non-200 status code returned: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] Error getting details for Place ID {place_id}: {e}")

    return {}

# -------------------------------------------------------------------
# CLASSIFICATION
# -------------------------------------------------------------------
def classify_service_type(types):
    """
    Classify the establishment based on the 'types' array from Google Places result.
    - BB1 -> Full service
    - BB2 -> Quick service
    - BB3 -> Bar
    - "Not a restaurant" if none of the above
    """
    print(f"[DEBUG] Classifying service type for types={types}")
    types_lower = [t.lower() for t in types]
    if "restaurant" in types_lower and "bar" not in types_lower and "fast_food" not in types_lower:
        return "BB1"  # Full service
    elif "fast_food" in types_lower or "meal_takeaway" in types_lower or "meal_delivery" in types_lower:
        return "BB2"  # Quick service
    elif "bar" in types_lower:
        return "BB3"  # Bar
    elif any(keyword in types_lower for keyword in ["cafe", "bakery", "food", "drink"]):
        # If it has "restaurant" as well, treat as full service
        return "BB1" if "restaurant" in types_lower else "BB2"
    else:
        return "Not a restaurant"

# -------------------------------------------------------------------
# SCRAPING HELPERS
# -------------------------------------------------------------------
def extract_emails(text):
    """
    Attempts to find email addresses in plain text using a simple pattern approach,
    with heuristics to capture local parts properly.
    """
    domain_pattern = r'@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}'
    matches = list(re.finditer(domain_pattern, text))

    cleaned_emails = set()
    known_usernames = ["info", "contact", "reservations", "sales", "support", "admin"]
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+-")

    for m in matches:
        domain_str = m.group(0)  # e.g. '@domain.com'
        domain = domain_str[1:]  # remove '@', e.g. 'domain.com'
        if not re.match(r'^[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$', domain):
            continue

        start_idx = m.start()
        pos = start_idx - 1
        local_chars = []
        while pos >= 0 and text[pos] in allowed_chars:
            local_chars.append(text[pos])
            pos -= 1
        if not local_chars:
            continue

        local_part = "".join(reversed(local_chars))

        found_username = False
        l_lower = local_part.lower()
        for uname in known_usernames:
            if l_lower.endswith(uname):
                idx = l_lower.rfind(uname)
                local_part = local_part[idx:]
                found_username = True
                break

        if not found_username:
            alpha_matches = re.findall(r'[A-Za-z]+', local_part)
            if alpha_matches:
                last_alpha = alpha_matches[-1]
                pos2 = local_part.lower().rfind(last_alpha.lower())
                local_part = local_part[pos2:]

        local_part = re.sub(r'^[^A-Za-z0-9]+', '', local_part)
        if not local_part:
            continue

        cleaned_email = local_part + '@' + domain
        cleaned_emails.add(cleaned_email)

    return cleaned_emails

def detect_reservation_platform(html_content):
    """
    Simple substring checks for Resy/OpenTable/Tock in the website's HTML.
    """
    if ('id="resy_button_container"' in html_content or 
        "widgets.resy.com" in html_content or 
        "resy.com" in html_content or 
        "Resy" in html_content):
        return "Resy"
    if "OpenTable" in html_content or "opentable.com" in html_content:
        return "OpenTable"
    if "Tock" in html_content or "exploretock.com" in html_content:
        return "Tock"
    return ""

def scrape_emails_and_pos_from_website(start_url, max_links=10):
    """
    Crawls up to `max_links` pages within the same domain to discover emails, POS systems, loyalty programs,
    and reservation platforms.
    """
    print(f"[DEBUG] Starting website scrape from: {start_url}")
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

        print(f"[DEBUG] Scraping URL: {url}")
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            html_content = soup.prettify()

            # Extract emails
            new_emails = extract_emails(text)
            # Clean up emails (trim anything after .com, etc.)
            cleaned_emails = set()
            for email in new_emails:
                if '@' in email and '.com' in email:
                    idx = email.find('.com')
                    cleaned_email = email[:idx+4]
                    cleaned_emails.add(cleaned_email)
                else:
                    cleaned_emails.add(email)

            emails_found.update(cleaned_emails)

            # Check for POS/loyalty references
            if "www.toasttab.com" in html_content and pos_system != "Toast":
                pos_system = "Toast"
            if "inkindscript.com" in html_content and "inKind" not in loyalty_programs:
                loyalty_programs.append("inKind")
            if "spoton.com" in html_content and "SpotOn" not in loyalty_programs:
                loyalty_programs.append("SpotOn")

            # Check reservation platform
            if not reservation_platform:
                platform = detect_reservation_platform(html_content)
                if platform:
                    reservation_platform = platform

            # Gather more links (internal only)
            priority_links = []
            normal_links = []
            for link in soup.find_all('a', href=True):
                absolute_link = urljoin(url, link['href'])
                parsed_link = urlparse(absolute_link)

                # Only scrape within the same domain to avoid drifting too far
                if parsed_link.netloc == base_domain:
                    # Skip big file types that won't have text
                    if parsed_link.path.endswith((".pdf", ".jpg", ".png")):
                        continue

                    link_text = link.get_text().lower()
                    link_url = absolute_link.lower()
                    if any(k in link_text or k in link_url for k in ["reservation", "book", "resy", "opentable", "tock"]):
                        priority_links.append(absolute_link)
                    else:
                        normal_links.append(absolute_link)

            # Add priority links first, then normal
            for pl in priority_links:
                if len(visited) + len(to_scrape) < max_links + 1:
                    to_scrape.append(pl)

            for nl in normal_links:
                if len(visited) + len(to_scrape) < max_links + 1:
                    to_scrape.append(nl)

            links_scraped += 1

        except requests.exceptions.RequestException as e:
            print(f"[DEBUG] Error scraping {url}: {e}")
            continue

    loyalty_programs = list(set(loyalty_programs))
    print(f"[DEBUG] Finished scraping. Found emails: {emails_found}, POS: {pos_system}, "
          f"Loyalty: {loyalty_programs}, Reservation: {reservation_platform}")
    return emails_found, pos_system, '; '.join(loyalty_programs), reservation_platform

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    if not API_KEY:
        print("[DEBUG] Missing MAPS_API_KEY environment variable. Exiting.")
        return

    # 1) Read the input CSV of restaurants
    try:
        df_input = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"[DEBUG] Could not find the file '{INPUT_FILE}'. Exiting.")
        return
    
    if "whole name" not in df_input.columns:
        print(f"[DEBUG] The CSV must contain a column named 'whole name'. Exiting.")
        return

    detailed_data = []

    # 2) Iterate over each restaurant name in the CSV
    for idx, row in df_input.iterrows():
        restaurant_name = str(row["whole name"]).strip()
        print(f"\n=== Processing {idx+1}/{len(df_input)}: {restaurant_name} ===")

        # Google Places: text search in NYC
        search_result = search_restaurant_in_nyc(restaurant_name, API_KEY)
        if not search_result:
            # No matches found
            entry = {
                "Input Name": restaurant_name,
                "Place Name": "",
                "Address": "",
                "Price Level": "",
                "Types": "",
                "Category": "Not found",
                "Website": "",
                "Phone Number": "",
                "Rating": "",
                "Review Count": "",
                "Opening Hours": "",
                "Email": "",
                "POS System": "",
                "Loyalty Programs": "",
                "Reservation Platform": "",
                "Review Text": "",
                "Review Author": "",
                "Review Rating": "",
                "Popular Dish/Drink": "[No data]",
                "Intro Email Blurb": "[No data]"
            }
            detailed_data.append(entry)
            continue

        # Extract the place_id from the search result
        place_id = search_result.get("place_id")
        if not place_id:
            print("[DEBUG] No place_id found, skipping.")
            continue

        # 3) Get place details
        details = get_place_details(place_id, API_KEY)
        if not details:
            print("[DEBUG] get_place_details returned empty, skipping.")
            continue

        # Basic fields
        name = details.get("name", "")
        address = details.get("formatted_address", "")
        price_level = details.get("price_level", "")
        types = details.get("types", [])
        website = details.get("website", "")
        phone = details.get("formatted_phone_number", "")
        rating = details.get("rating", "")
        user_ratings_total = details.get("user_ratings_total", "")
        opening_hours_raw = details.get("opening_hours", {}).get("weekday_text", [])
        opening_hours = ", ".join(opening_hours_raw)

        # Classify as bar, restaurant, etc.
        service_type = classify_service_type(types)

        # Best single review
        top_review = details.get("most_relevant_review")
        if top_review:
            review_text = top_review.get("text", "")
            review_author = top_review.get("author_name", "")
            review_rating = top_review.get("rating", "")
        else:
            review_text = ""
            review_author = ""
            review_rating = ""

        # All reviews, for the AI calls
        all_reviews = details.get("reviews", [])

        # 4) AI: popular dish/drink
        popular_dish = find_popular_dish(name, all_reviews)

        # AI: short email intro
        intro_blurb = generate_intro(name, all_reviews, popular_dish)

        # 5) Website scraping for emails, POS, etc.
        if website:
            emails, pos_system, loyalty_programs, reservation_platform = scrape_emails_and_pos_from_website(website, max_links=10)
        else:
            emails, pos_system, loyalty_programs, reservation_platform = set(), "", "", ""

        # If no emails found, store an empty
        if not emails:
            emails = [""]

        # Build final output rows. If multiple emails, create multiple rows.
        for email in emails:
            entry = {
                "Input Name": restaurant_name,
                "Place Name": name,
                "Address": address,
                "Price Level": price_level,
                "Types": ", ".join(types),
                "Category": service_type,
                "Website": website,
                "Phone Number": phone,
                "Rating": rating,
                "Review Count": user_ratings_total,
                "Opening Hours": opening_hours,
                "Email": email,
                "POS System": pos_system,
                "Loyalty Programs": loyalty_programs,
                "Reservation Platform": reservation_platform,
                "Review Text": review_text,
                "Review Author": review_author,
                "Review Rating": review_rating,
                "Popular Dish/Drink": popular_dish,
                "Intro Email Blurb": intro_blurb
            }
            detailed_data.append(entry)

    # 6) Save results to CSV
    if detailed_data:
        df_output = pd.DataFrame(detailed_data)
        df_output.to_csv(OUTPUT_FILE, index=False)
        print(f"[DEBUG] Done! Data saved to {OUTPUT_FILE}")
    else:
        print("[DEBUG] No data to save.")

if __name__ == "__main__":
    main()
