import time
import torch
import base64
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

PROMPT = """
#Your Task: Receipt Recognition and Data Extraction

You are tasked with extracting structured information from receipts. Receipts will come from various countries, in different languages, and can have different layouts and formats. Your goal is to parse the receipt text, identify the receipt type (store, cafe/restaurant, or payment for services), and return the data in JSON format with the required fields. Follow the specific instructions below to ensure accurate extraction.

#Required Fields:

1. Receipt Type: Identify the type of receipt. It could be from:
	* Store: Typically involves grocery or retail items.
	* Cafe/Restaurant: Typically involves food and beverage items, table numbers, or tipping sections.
	* Payment for Services: This type of receipt may involve service fees or professional services.
2. Receipt Number: Extract the unique receipt number, typically found at the top of the receipt.
3. Store/Business Name: Extract the name of the store, cafe, restaurant, or service provider.
4. Store Address: Extract the address of the store, including city and country if available.
5. Date: Extract the date of the transaction and format it as YYYY-MM-DD HH:MM.
6. Currency: Extract the currency if explicitly mentioned (e.g., EUR, USD). If the currency is not specified, detect the language of the receipt and infer the currency based on the country where the language is predominantly used. For example, if the receipt is in Ukrainian, set the currency to UAH (Ukrainian Hryvnia).
7. Payment Method: Identify whether the payment was made by "card" or "cash."
8. Total Amount: Extract the total amount of the transaction. This is typically located at the end of the receipt, often highlighted in bold or a larger font.
9. Total Discount: Extract the total discount if explicitly mentioned. If not, calculate the total discount by summing up the discounts for individual items.
10. Tax: Extract the total tax amount if it is listed on the receipt.

#Item-Level Details:

For each item on the receipt, extract the following details:

1. Item Name: Extract the full name of each item. Some items may have names split across multiple lines; in this case, concatenate the lines until you encounter a quantity or unit of measurement (e.g., "2ks"), which marks the end of the item name or on the next line. You should extract full name till statements as, for example, "1 ks" or "1 ks * 2"
2. Unit Price: Extract the price per unit for each item.
3. Quantity: Extract the quantity of each item, including the unit of measurement (e.g., "ks" for pieces, "kg" for kilograms).
4. Price: Extract the final price for each item.
5. Discount: Extract any discount applied to the item. If no discount is provided, set it to 0.
6. Category: Automatically assign a category based on the item name. For groceries, assign relevant subcategories such as Dairy, Bakery, Fruits, etc. If this receipt was from restaurant - you should put category only from this list: Food, Drinks.

#Special Cases:

1. Cafe/Restaurant Receipts: If the receipt is from a cafe or restaurant, handle additional fields like:
	* Table Number: Extract the table number if available, often printed near the top of the receipt.
	* Tips: Extract any tip amounts explicitly listed or infer from the total paid amount minus the original bill amount.
	* Service Charges: Some restaurants may include an automatic service charge, which should be listed separately from the tax or tips.
	* Order Type: Identify whether the order was "dine-in" or "takeaway."
2. Missing Currency: If no currency is mentioned on the receipt, infer the local currency by detecting the language and country of origin. For example, a receipt in French would use EUR, while one in Ukrainian would use UAH.
3. Multi-line Item Names: If an item name spans multiple lines, merge the lines to form the complete name. Stop merging when a quantity or unit of measurement is encountered.
4. Total Amount: The total amount is often larger than other numbers or displayed in bold at the bottom of the receipt. Make sure to capture this accurately.
5. Total Discount: If no total discount is listed, sum the discounts for each individual item.
6. Rounding Adjustments: Some receipts may include a "rounding" line item, where the total amount is adjusted (typically for cash payments) to avoid dealing with fractions of currency (e.g., rounding to the nearest 0.05 in some countries). If a rounding adjustment is present, extract the value of the rounding adjustment and reflect it in the total amount. For example:
	* Total Before Rounding: 19.97
	* Rounding: -0.02
	* Final Total: 19.95 If the rounding adjustment is found, include it as a separate field in the JSON output under "rounding_adjustment", and ensure that the "total_amount" reflects the final adjusted total.
7. Taxes: Receipts can handle taxes in various ways, and the system should be prepared to capture these scenarios:
	* Tax-Inclusive Pricing: In some countries or for certain receipts, taxes are already included in the item price and not listed separately. If the receipt mentions that taxes are included in prices, record the "tax" field as 0 and note that taxes are included in the item prices.
	* Multiple Tax Rates: Some receipts may include multiple tax rates (e.g., different VAT rates for different items). In this case, extract each tax rate and the corresponding tax amounts, and store them in a separate list of tax breakdowns. For example, the receipt might show "5% VAT" and "15% VAT" for different categories of goods:
		** "taxes": [{"rate": "5%", "amount": 1.00}, {"rate": "15%", "amount": 3.50}]
	* Missing Tax Information: In some cases, the receipt might not clearly mention taxes, but you may infer them based on standard rates in the country of origin. If no explicit tax amount is listed and you are unable to infer it, set the tax to "unknown" or null in the JSON output.
	* Tax-Exempt Items: Some items on the receipt may be tax-exempt. If this is indicated, ensure that these items are excluded from any tax calculations. Note these in the item-level details with "tax_exempt": true and make sure the "tax" field reflects the correct amount for taxable items only.
	* Service Charges vs. Taxes: Sometimes service charges may be listed separately from taxes (common in restaurants). Ensure that service charges are not included in the tax amount, and store them under the "service_charge" field.
	* Tax Breakdown and Total: If both individual item taxes and a total tax amount are listed, the system should ensure consistency between the sum of item-level taxes and the total tax listed at the bottom of the receipt.
8. In certain receipt formats, the quantity and unit price may appear before the item name. When processing such receipts, the goal is to correctly extract the quantity, unit price, and item name in their proper order. For example, if one line of the receipt shows "5 * 23.00 = 115.0" and the next line displays "Milk," the system should interpret this as:
	* Quantity: 5 units
	* Unit Price: 23.00
	* Item Name: Milk
	* Total Price: 115.0 This approach should be applied consistently throughout the entire receipt to extract data accurately.

#JSON Output Format:

{
  "receipt_type": "string",
  "receipt_number": "string",
  "store_name": "string",
  "store_address": "string",
  "date_time": "string",
  "currency": "string",
  "payment_method": "string",
  "total_amount": "number",
  "total_discount": "number",
  "tax": "number",
  "taxes": [
    {
      "rate": "string",
      "amount": "number"
    }
  ],
  "rounding_adjustment": "number",
  "rounded_total_aount": "number",
  "items": [
    {
      "name": "string",
      "unit_price": "number",
      "quantity": {
        "amount": "number",
        "unit_of_measurement": "string"
      },
      "price": "number",
      "discount": "number",
      "category": "string",
      "tax_exempt": "boolean"
    }
  ],
  "cafe_additional_info": {
    "table_number": "string",
    "tips": "number",
    "service_charge": "number",
    "order_type": "string"
  }
}

#Additional Notes:

1. You should handle receipts in various languages and from different countries.
2. Pay special attention to formatting differences and edge cases, such as multi-line item names, missing currency symbols, or cafe/restaurant-specific information.
3. Always ensure the output is well-structured and follows the JSON format provided.
4. The "rounding_adjustment" field should reflect the value by which the total was adjusted due to rounding. If no rounding adjustment is present, it can be set to 0 or omitted from the output.
5. Ensure that the final "total_amount" field reflects the total after any rounding adjustment has been applied.
6. Inclusive Taxes: If taxes are included in the item prices, set the "tax" field to 0 and adjust the item prices accordingly.
7. Multiple Tax Rates: The "taxes" field provides a detailed breakdown for receipts with different tax rates. This field is optional and can be excluded if only a single tax amount is listed.
8. Tax-Exempt Items: Mark tax-exempt items with the "tax_exempt": true field.
9. Service Charges vs. Taxes: Ensure that service charges are captured separately from taxes in the "service_charge" field.
10. Return the full JSON object with all available information. If any information is unclear or missing, include it as "unknown" or "not available" in the output.
11. Your final response should be in valid JSON format with no additional text.
"""

# Choose device: Use MPS if available, otherwise CUDA or CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
# Load the model (ensure you have an updated version of transformers)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto",
    # Uncomment the next line if flash_attention_2 is supported
    # attn_implementation="flash_attention_2"
)
# Move model to the selected device if not already assigned by device_map.
model.to(device)

processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

image_path = "archive/images/8.jpg"  # Replace with your image file path
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
# Prepend the base64 string with the data URI scheme for JPEG images
base64_image = f"data:image/jpeg;base64,{encoded_image}"

# Create the message with the base64 image and a text prompt
data = {
    "messages": [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": base64_image}
        ]}
    ]
}

messages = data.get("messages", [])
if not messages:
    print({"error": "No messages provided."})

print("Building the prompt...")
# Build the prompt and process image inputs
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

print("Starting to generate...")
start_time = time.time()

# Generate response
generated_ids = model.generate(**inputs, max_new_tokens=1024)

end_time = time.time()
print(f"Generation finished in {end_time - start_time:.2f} seconds.")

# Trim the input tokens from the output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

print("Processing output text...")
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(output_text)