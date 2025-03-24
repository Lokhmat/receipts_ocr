import os
import requests
from together import Together
import json

class JsonParser:
    def __init__(self):
        # Initialize Together client with API key from environment
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is not set")
        self.client = Together(api_key=api_key)
        
        # The same prompt as before
        self.PROMPT = """
        #Your Task: Receipt Recognition and Data Extraction

You are tasked with extracting structured information from receipts. Receipts will come from various countries, in different languages, and can have different layouts and formats. Your goal is to parse the receipt text and return the data in JSON format with the required fields. Follow the specific instructions below to ensure accurate extraction.

#Required Fields:

1. Receipt Number: Extract the unique receipt number, typically found at the top of the receipt.
2. Store/Business Name: Extract the name of the store, cafe, restaurant, or service provider.
3. Store Address: Extract the address of the store, including city and country if available.
4. Date: Extract the date of the transaction and format it as YYYY-MM-DD HH:MM.
5. Currency: Extract the currency if explicitly mentioned (e.g., EUR, USD). If the currency is not specified, leave it as null.
6. Total Amount: Extract the total amount of the transaction. This is typically located at the end of the receipt, often highlighted in bold or a larger font.
7. Total Discount: Extract the total discount if explicitly mentioned. If not, calculate the total discount by summing up the discounts for individual items.
8. Tax: Extract the total tax amount if it is listed on the receipt.

#Item-Level Details:

For each item on the receipt, extract the following details:

1. Item Name: Extract the full name of each item. Some items may have names split across multiple lines; in this case, concatenate the lines until you encounter a quantity or unit of measurement (e.g., "2kg"), which marks the end of the item name or indicates that the item details have begun.
2. Quantity: Extract the quantity of each item, taking into account both the numerical amount and the unit of measurement.
3. Price: Extract the final price for each item or position on the receipt.
4. Discount: Optionally extract any discount associated with the item, if available.

#Handling Quantity and Units:

When extracting quantity details, consider the following cases to correctly interpret items sold by weight versus items sold by piece:

1. **Items Sold by Weight:**
   - **Explicit Weight Information:** If the receipt shows a numerical value with a weight unit (such as "kg", "g", or "lb"), extract that number as the weight amount and the corresponding unit as the unit of measurement.
     - *Example:* For a line "Sugar 0.5 kg", extract `"quantity": { "amount": 0.5, "unit_of_measurement": "kg" }`.
   - **Weight Unit Variations:** Recognize variations in units (e.g., "kgs", "kilograms", "grams", "lbs", "pounds"). Normalize these into one of the supported units: `kg`, `g`, or `lb`.
   - **Multi-line Formats:** If weight information is split over multiple lines or appears alongside price details (for example, "Bananas" on one line and "1.2 kg" on the next), merge the lines to correctly assign the weight amount and unit.

2. **Items Sold by Piece:**
   - **Explicit Piece Information:** If the receipt indicates the number of pieces (often shown as "pcs" or implied by a multiplication format such as "5 * 23.00 = 115.0"), assign the unit of measurement as `"pcs"`.
     - *Example:* For a line "Eggs 12 pcs", extract `"quantity": { "amount": 12, "unit_of_measurement": "pcs" }`.
   - **Multiplication Format Cases:** When a line such as "5 * 23.00 = 115.0" appears, interpret it as 5 pieces, and look for the corresponding item name in an adjacent line. Ensure that `"quantity": { "amount": 5, "unit_of_measurement": "pcs" }` is captured.
   - **Inconsistent or Missing Quantity Information:** If the receipt implies that an item is sold by pieces (for example, through context or typical product type) but does not clearly provide a quantity, default to 1. This means if the quantity is ambiguous or absent, assume that the client bought 1 unit.

3. **Ambiguous or Inconsistent Cases:**
   - **Missing or Unclear Units:** If a receipt does not clearly specify the unit of measurement but the context implies weight (for example, by showing decimal numbers or using terms like “kg” in nearby text), extract the weight if it is clear; otherwise, mark the unit as `"not available"` and the amount as `"unknown"`.
   - **Context-Dependent Interpretation:** In cases where the same receipt might list both weighted items and items sold by piece, use contextual clues (such as typical product types or formatting cues) to determine whether the number represents a weight or a count. If ambiguous, prioritize explicitly stated units; if none are provided and the context suggests pieces, default the quantity to 1.

#Special Cases:

1. Multi-line Item Names: If an item name spans multiple lines, merge the lines to form the complete name. Stop merging when a quantity or unit of measurement is encountered.
2. Total Amount: The total amount is often larger than other numbers or displayed in bold at the bottom of the receipt. Make sure to capture this accurately.
3. Total Discount: If no total discount is listed, sum the discounts for each individual item.
4. Total Tax: Find the total tax amount that is paid or should be paid. It is usually found at the bottom of the receipt.
5. Quantity Details: The quantity of an item might appear before, after, or on the same line as the item name. Use spatial and contextual cues to merge the relevant information. For example:
   - If one line shows "5 * 23.00 = 115.0" and the next line shows "Milk," interpret this as 5 pieces of Milk at a total price of 115.0.
   - Conversely, if the item name appears first (e.g., "Milk") and a following line shows "5 * 23.00 = 115.0," treat it the same way.

#JSON Output Format:

{
  "receipt_number": "string",
  "store_name": "string",
  "store_address": "string",
  "date_time": "string",
  "currency": "optional[string]",
  "total_amount": "number",
  "total_discount": "number",
  "total_tax": "number",
  "items": [
    {
      "name": "string",
      "quantity": {
        "amount": "number",
        "unit_of_measurement": "enumeration[pcs, kg, lb, g]"
      },
      "price": "number",
      "discount": "optional[number]"
    }
  ]
}

#Additional Notes:

1. Handle receipts in various languages and from different countries.
2. Pay special attention to formatting differences and edge cases, such as multi-line item names, missing currency symbols, and variations in quantity placement.
3. **Handling Units:**
   - For items sold by weight, extract the numerical weight and normalize the unit to one of the supported values (`kg`, `g`, or `lb`).
   - For items sold by piece, ensure that the unit is set to `"pcs"` and that the number accurately reflects the quantity purchased.
   - When faced with ambiguous formatting or missing quantity information for items that appear to be sold by pieces, default the quantity to 1.
4. Always ensure the output is well-structured and follows the JSON format provided.
5. Return the full JSON object with all available information. If any information is unclear or missing, include it as `"unknown"` or `"not available"` in the output.
6. Do not add any additional text outside the JSON output.
        """

    def _upload_to_tmpfiles(self, image_data: bytes) -> str:
        """Upload image to tmpfiles.org and return the download URL"""
        try:
            # Upload to tmpfiles.org
            files = {'file': ('receipt.jpg', image_data)}
            response = requests.post('https://tmpfiles.org/api/v1/upload', files=files)
            response.raise_for_status()
            
            # Parse response to get file ID
            upload_data = response.json()
            file_url = upload_data['data']['url'].replace('tmpfiles.org', 'tmpfiles.org/dl')
            
            # Construct direct download URL
            print(file_url)
            return file_url
        except Exception as e:
            raise Exception(f"Failed to upload image to tmpfiles.org: {e}")

    def parse_image(self, image_data: bytes) -> dict:
        try:
            # Upload image to tmpfiles.org
            image_url = self._upload_to_tmpfiles(image_data)
            
            # Create chat completion with Together.ai
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2-VL-72B-Instruct",
                messages=[
                    {"role": "system", "content": self.PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]}
                ]
            )
            
            # Extract and parse JSON from response
            json_text = response.choices[0].message.content
            json_text = json_text.replace("```json", "").replace("```", "")
            return json.loads(json_text)
            
        except Exception as e:
            print(f"Failed to process image: {e}")
            return {}
