import pandas as pd
import requests
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import os
import re

# Anthropic / Claude
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from anthropic import InternalServerError

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
API_KEY = os.environ.get("MAPS_API_KEY")        # Replace with your actual Google Maps API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # Claude API key (if available)
OUTPUT_FILE = "categorized_restaurants_with_details.csv"  # Output CSV file

FIND_PLACE_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# Create a session for performance and reusability
session = requests.Session()

# -------------------------------------------------------------------
# AI HELPER
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

    # Prompt: ask only for the top-mentioned dish or drink
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
        # We will just return the entire completion (should ideally be just the dish name)
        return completion
    except InternalServerError as e:
        print(f"[DEBUG] Error calling Claude: {e}")
        return "[AI Error]"

def generate_intro_and_times(restaurant_name, reviews, popular_dish, opening_hours):
    """
    Generate a short "intro email blurb" referencing the popular dish and 
    positive ambiance/staff/vibe from the reviews. Then suggest a couple 
    of times to drop in when it's likely less busy, based on the restaurant's 
    opening hours. Phase this something like "Could Michelle, our head of SF, stop by at [day and time] or [day and time option2], I assume you may be slightly less busy then?"

    Returns (intro_blurb, suggested_times).
    If ANTHROPIC_API_KEY is missing, returns placeholder text.
    """
    if not ANTHROPIC_API_KEY:
        return ("[Missing Anthropic Key for Intro]", "[N/A]")

    # Combine up to 5 reviews for context
    relevant_reviews = "\n\n".join(r.get("text", "") for r in reviews[:5]).strip()
    # If we have no opening hours, pass a placeholder
    hours_text = "\n".join(opening_hours.get("weekday_text", [])) if opening_hours else "[No hours data]"

    # We'll ask Claude to produce EXACTLY two paragraphs:
    # Paragraph #1: Intro blurb
    # Paragraph #2: Suggested times
    prompt = (
        f"{HUMAN_PROMPT}\n"
        "You are given:\n"
        f"- Restaurant name: {restaurant_name}\n"
        f"- A 'popular dish': {popular_dish}\n"
        " - Up to 5 recent reviews (for vibe, ambiance, staff, etc.):\n"
        f"{relevant_reviews}\n\n"
        " - Opening hours:\n"
        f"{hours_text}\n\n"
        "Task:\n"
        "1) Write a short personal-sounding sentence or two for an intro to a prospecting email, referencing the popular dish and something positive (ambiance, staff, vibe, etc.) gleaned from the reviews. For example:\n"
        "   'I stopped in a couple weeks ago with my friends and we ordered the [dish] - it was so delicious, and the [ambiance/staff/vibe] was so [cozy/welcoming]. We had the best time!'\n\n"
        "2) Based on the opening hours, suggest a day/time or two for our head of SF to drop in when the restaurant is likely not as busy (e.g. after lunch rush, or just before dinner, or any relevant quieter window).\n\n"
        "Return your response in two paragraphs:\n"
        "Paragraph #1 => The short intro blurb\n"
        "Paragraph #2 => Propose two daes and times for that Michelle, our head of SF, can stop in.\n\n"
        "No extra commentary.\n"
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

        if "\n\n" in completion:
            intro, times = completion.split("\n\n", 1)
            return (intro.strip(), times.strip())
        else:
            # If we can’t split, treat everything as intro only
            return (completion, "[Could not parse times]")
    except InternalServerError as e:
        print(f"[DEBUG] Error calling Claude for intro & times: {e}")
        return ("[AI Error]", "[N/A]")

# -------------------------------------------------------------------
# ORIGINAL FUNCTIONS (minus poem references)
# -------------------------------------------------------------------
def find_place_id(restaurant_name, api_key):
    print(f"\n[DEBUG] Attempting to find place ID for: '{restaurant_name}'")
    print(f"[DEBUG] API key in find_place_id is: {api_key}")

    params = {
        "input": restaurant_name,
        "inputtype": "textquery",
        "fields": "place_id",
        "key": api_key,
        # SF location bias
        "locationbias": "circle:20000@37.7749,-122.4194"
        # NYC location bias
        # "locationbias": "circle:20000@40.7128,74.0060"
    }
    print(f"[DEBUG] Params for FIND_PLACE_URL: {params}")

    try:
        response = session.get(FIND_PLACE_URL, params=params, timeout=10)
        print(f"[DEBUG] Response status_code: {response.status_code}")
        if response.status_code == 200:
            print(f"[DEBUG] Raw JSON from find_place_id:\n{response.text}\n")
            result = response.json().get("candidates", [])
            if result:
                print(f"[DEBUG] Found place_id: {result[0].get('place_id')}")
                return result[0].get("place_id")
            else:
                print("[DEBUG] No candidates returned in the JSON.")
        else:
            print(f"[DEBUG] Non-200 status code returned: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] Error finding place ID for {restaurant_name}: {e}")
    return None

def get_place_details(place_id, api_key):
    print(f"\n[DEBUG] Getting details for Place ID: '{place_id}'")
    print(f"[DEBUG] API key in get_place_details is: {api_key}")

    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,price_level,types,website,formatted_phone_number,"
                  "rating,user_ratings_total,opening_hours,reviews",
        "key": api_key
    }
    print(f"[DEBUG] Params for PLACE_DETAILS_URL: {params}")

    try:
        response = session.get(PLACE_DETAILS_URL, params=params, timeout=10)
        print(f"[DEBUG] Response status_code: {response.status_code}")
        if response.status_code == 200:
            print(f"[DEBUG] Raw JSON from get_place_details:\n{response.text}\n")
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

def classify_service_type(types):
    print(f"[DEBUG] Classifying service type for types={types}")
    types = [t.lower() for t in types]
    if "restaurant" in types and "bar" not in types and "fast_food" not in types:
        return "BB1"  # Full service
    elif "fast_food" in types or "meal_takeaway" in types or "meal_delivery" in types:
        return "BB2"  # Quick service
    elif "bar" in types:
        return "BB3"  # Bar
    elif any(keyword in types for keyword in ["cafe", "bakery", "food", "drink"]):
        return "BB2" if "restaurant" not in types else "BB1"
    else:
        return "Not a restaurant"

def clean_name(restaurant_name):
    cleaned = restaurant_name.replace("greater SF area", "").strip()
    return f"{cleaned}"

def extract_emails(text):
    domain_pattern = r'@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}'
    matches = list(re.finditer(domain_pattern, text))

    cleaned_emails = set()
    known_usernames = ["info", "contact", "reservations", "sales", "support", "admin"]
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+-")

    for m in matches:
        domain_str = m.group(0)  # includes '@'
        domain = domain_str[1:]  # remove '@'
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
                pos = local_part.lower().rfind(last_alpha.lower())
                local_part = local_part[pos:]

        local_part = re.sub(r'^[^A-Za-z0-9]+', '', local_part)
        if not local_part:
            continue

        cleaned_email = local_part + '@' + domain
        cleaned_emails.add(cleaned_email)

    return cleaned_emails

def detect_reservation_platform(html_content):
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

            if not reservation_platform:
                platform = detect_reservation_platform(html_content)
                if platform:
                    reservation_platform = platform

            # Gather more links
            priority_links = []
            normal_links = []
            for link in soup.find_all('a', href=True):
                absolute_link = urljoin(url, link['href'])
                parsed_link = urlparse(absolute_link)

                if parsed_link.netloc == base_domain:
                    # Skip big filetypes
                    if parsed_link.path.endswith((".pdf", ".jpg", ".png")):
                        continue

                    link_text = link.get_text().lower()
                    link_url = absolute_link.lower()
                    if any(k in link_text or k in link_url for k in ["reservation", "book", "resy", "opentable", "tock"]):
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
            print(f"[DEBUG] Error scraping {url}: {e}")
            continue

    loyalty_programs = list(set(loyalty_programs))
    print(f"[DEBUG] Finished scraping. Found emails: {emails_found}, POS: {pos_system}, Loyalty: {loyalty_programs}, Reservation: {reservation_platform}")
    return emails_found, pos_system, '; '.join(loyalty_programs), reservation_platform

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main(input_file):
    print(f"[DEBUG] Loading input file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"[DEBUG] Number of rows in input file: {len(df)}")
    print(f"[DEBUG] Columns: {df.columns.tolist()}")

    if "whole name" not in df.columns:
        print("[DEBUG] Error: 'whole name' column not found.")
        return

    if df.empty:
        print("[DEBUG] The input CSV is empty. No restaurants to process.")
        return

    detailed_data = []
    try:
        existing_df = pd.read_csv(OUTPUT_FILE)
        processed_names = set(existing_df["whole name"].unique())
        detailed_data = existing_df.to_dict('records')
        print(f"[DEBUG] Loaded existing progress with {len(existing_df)} records.")
    except FileNotFoundError:
        processed_names = set()

    for index, row in df.iterrows():
        restaurant_name = clean_name(row["whole name"])
        if restaurant_name in processed_names:
            print(f"[DEBUG] Already processed {restaurant_name}, skipping.")
            continue

        print(f"\n[DEBUG] Processing: {restaurant_name}")

        # Step 1: Find Place ID
        place_id = find_place_id(restaurant_name, API_KEY)
        if not place_id:
            print(f"[DEBUG] No Place ID found for {restaurant_name}. Saving progress and continuing.")
            pd.DataFrame(detailed_data).to_csv(OUTPUT_FILE, index=False)
            continue

        # Step 2: Get Place Details
        details = get_place_details(place_id, API_KEY)
        if not details:
            print(f"[DEBUG] No details found for Place ID {place_id} ({restaurant_name}). Saving progress and continuing.")
            pd.DataFrame(detailed_data).to_csv(OUTPUT_FILE, index=False)
            continue

        # Log
        print(f"[DEBUG] Google returned name: {details.get('name')}")
        print(f"[DEBUG] Google returned website: {details.get('website')}")

        # Step 3: Classify Service Type
        service_type = classify_service_type(details.get("types", []))

        # Step 4: Extract the "most relevant" review
        review = details.get("most_relevant_review", {})
        review_text = review.get("text", "") if review else "No reviews available"
        review_author = review.get("author_name", "") if review else "N/A"
        review_rating = review.get("rating", "") if review else "N/A"

        # Step 5: Collect all reviews for AI
        all_reviews = details.get("reviews", [])

        # Generate the popular dish only (no poem)
        popular_dish = find_popular_dish(restaurant_name, all_reviews)

        # Generate intro + suggested times
        intro_blurb, suggested_times = generate_intro_and_times(
            restaurant_name,
            all_reviews,
            popular_dish,
            details.get("opening_hours")
        )

        # Step 6: Optional website scraping
        website = details.get("website")
        if website:
            emails, pos_system, loyalty_programs, reservation_platform = scrape_emails_and_pos_from_website(website, max_links=10)
        else:
            emails, pos_system, loyalty_programs, reservation_platform = set(), "", "", ""

        if not emails:
            emails = [""]

        # Step 7: Append everything
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
                "Reservation Platform": reservation_platform,
                # Existing review columns
                "Review Text": review_text,
                "Review Author": review_author,
                "Review Rating": review_rating,
                # NEW columns
                "Popular Dish/Drink": popular_dish,
                "Intro Email Blurb": intro_blurb,
                "Suggested Times": suggested_times
            }
            detailed_data.append(entry)

        processed_names.add(restaurant_name)
        pd.DataFrame(detailed_data).to_csv(OUTPUT_FILE, index=False)
        print(f"[DEBUG] Progress saved after processing {restaurant_name}.")

    print(f"[DEBUG] All done. Data saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    input_file = "input.csv"  # Replace 'input.csv' with your actual input file name
    main(input_file)
