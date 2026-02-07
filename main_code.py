import cv2
import boto3
import json
import base64
import time
import re
from datetime import datetime
from flask import Flask, request

INVENTORY_FILE = 'smart_pantry_inventory.json'

app = Flask(__name__)


def load_aws_config():
    try:
        with open('aws_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_bedrock_client():
    config = load_aws_config()
    
    if config:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name=config.get('region', 'us-east-1'),
            aws_access_key_id=config['aws_access_key_id'],
            aws_secret_access_key=config['aws_secret_access_key']
        )
    else:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )


def load_inventory():
    try:
        with open(INVENTORY_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_inventory(inventory):
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(inventory, f, indent=2)


def compare_inventory(old_inv, new_inv):
    changes = {
        'added': {},
        'removed': {},
        'zero_items': []
    }
    
    all_items = set(list(old_inv.keys()) + list(new_inv.keys()))
    
    for item in all_items:
        old_count = old_inv.get(item, 0)
        new_count = new_inv.get(item, 0)
        
        if new_count > old_count:
            changes['added'][item] = new_count - old_count
        elif new_count < old_count:
            changes['removed'][item] = old_count - new_count
        
        if new_count == 0:
            changes['zero_items'].append(item)
    
    return changes


def capture_snapshot(camera_index=0, countdown=3):
    if countdown > 0:
        print(f"Capture countdown:")
        for i in range(countdown, 0, -1):
            print(f"{i}...")
            time.sleep(1)
    
    print("Opening camera...")
    cam = cv2.VideoCapture(camera_index)
    
    print("Initializing camera hardware...")
    time.sleep(2.0)
    
    print("Adjusting exposure...")
    for i in range(10):
        ret, frame = cam.read()
        time.sleep(0.1)
    
    print("Capturing image...")
    ret, frame = cam.read()
    cam.release()
    print("Image captured successfully.\n")
    
    return frame if ret else None


def image_to_base64(frame):
    success, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8') if success else None


def detect_items_with_bedrock(frame, client, mode='general'):
    image_base64 = image_to_base64(frame)
    if not image_base64:
        return None
    
    if mode == 'pantry':
        prompt = """Analyze this pantry/kitchen shelf. Identify and count ONLY food and drink items.

RULES:
- Only count actual pantry items (food, drinks, packages)
- Ignore furniture, decorations, people, clothing
- Be specific: "Coca-Cola can", "Red apple", "Oreo cookies"

Return ONLY this JSON:
{
  "items": [
    {"name": "specific item name", "count": number}
  ]
}"""
    else:
        prompt = """Look at this image and identify ALL distinct objects.

Count each separate object. Be specific with names.

Return ONLY this JSON:
{
  "items": [
    {"name": "specific object name", "count": number}
  ]
}"""
    
    try:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        }
        
        model_ids = [
            'anthropic.claude-3-sonnet-20240229-v1:0',
            'us.anthropic.claude-3-sonnet-20240229-v1:0',
            'anthropic.claude-3-haiku-20240307-v1:0',
        ]
        
        response = None
        for model_id in model_ids:
            try:
                response = client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body)
                )
                break
            except Exception as e:
                if "ValidationException" in str(e):
                    continue
                else:
                    raise
        
        if response is None:
            return None
        
        raw_body = response['body'].read()
        response_body = json.loads(raw_body)
        response_text = response_body['content'][0]['text']
        
        json_match = re.search(r'\{[\s\S]*"items"[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response_text
        
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        
        result = json.loads(json_str.strip())
        
        inventory = {}
        for item in result.get('items', []):
            inventory[item['name']] = item['count']
        
        return inventory
        
    except Exception as e:
        print(f"Detection error: {e}")
        return None


def process_scan(client, mode, camera_index, countdown):
    print(f"\n--- SCAN INITIATED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    old_inventory = load_inventory()
    if old_inventory:
        print(f"Previous inventory: {len(old_inventory)} item types")
    else:
        print("Previous inventory: empty")
    
    print("\nInitiating capture sequence...")
    frame = capture_snapshot(camera_index, countdown)
    
    if frame is None:
        print("ERROR: Camera capture failed")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = f"snapshot_{timestamp}.jpg"
    cv2.imwrite(snapshot_file, frame)
    print(f"Snapshot saved: {snapshot_file}")
    
    print("Running AI detection...")
    detected_inventory = detect_items_with_bedrock(frame, client, mode)
    
    if detected_inventory is None:
        print("ERROR: Detection failed")
        return
    
    # Merge with old inventory - keep items at zero
    new_inventory = {}
    
    # Start with all items from old inventory
    for item in old_inventory:
        new_inventory[item] = 0  # Default to zero
    
    # Update with detected items
    for item, count in detected_inventory.items():
        new_inventory[item] = count
    
    print(f"\nDetection complete:")
    for item, count in new_inventory.items():
        if count > 0:
            print(f"{item}: {count}")
    
    # Show zero items separately
    zero_items = [item for item, count in new_inventory.items() if count == 0]
    if zero_items:
        print(f"\nItems at zero:")
        for item in zero_items:
            print(f"{item}: 0")
    
    changes = compare_inventory(old_inventory, new_inventory)
    
    if changes['added']:
        print("\nItems added:")
        for item, count in changes['added'].items():
            print(f"+{count} {item}")
    
    if changes['removed']:
        print("\nItems removed:")
        for item, count in changes['removed'].items():
            print(f"-{count} {item}")
    
    save_inventory(new_inventory)
    print(f"\nInventory updated\n")


def show_inventory():
    inventory = load_inventory()
    
    if not inventory:
        print("\nInventory is empty. Run a scan first.\n")
        return
    
    print(f"\n--- CURRENT INVENTORY ({len(inventory)} items) ---")
    
    zero_items = []
    active_items = []
    
    for item, count in sorted(inventory.items()):
        if count == 0:
            zero_items.append(item)
        else:
            active_items.append((item, count))
    
    if active_items:
        print("\nIn stock:")
        for item, count in active_items:
            print(f"{item}: {count}")
    
    if zero_items:
        print("\nOut of stock:")
        for item in zero_items:
            print(f"{item}: 0")
    
    print()


@app.route('/whatsapp', methods=['GET'])
def whatsapp_test():
    return 'WhatsApp webhook is running!', 200


@app.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    from twilio.twiml.messaging_response import MessagingResponse
    from twilio.request_validator import RequestValidator
    
    print(f"[WEBHOOK] Received request")
    print(f"[WEBHOOK] From: {request.values.get('From', 'unknown')}")
    print(f"[WEBHOOK] Message: {request.values.get('Body', '')}")
    print(f"[WEBHOOK] Request URL: {request.url}")
    
    # Load Twilio config for validation
    try:
        with open('twilio_config.json', 'r') as f:
            config = json.load(f)
        
        # Validate request signature
        validator = RequestValidator(config['auth_token'])
        
        # When using ngrok, we need to use the forwarded URL
        url = request.url
        if request.headers.get('X-Forwarded-Proto'):
            # Reconstruct the actual URL Twilio called
            proto = request.headers.get('X-Forwarded-Proto', 'https')
            host = request.headers.get('X-Forwarded-Host') or request.headers.get('Host')
            url = f"{proto}://{host}{request.path}"
            print(f"[WEBHOOK] Using ngrok URL for validation: {url}")
        
        params = request.form.to_dict()
        signature = request.headers.get('X-Twilio-Signature', '')
        
        if not validator.validate(url, params, signature):
            print("[WEBHOOK] Invalid signature!")
            print(f"[WEBHOOK] Expected URL: {url}")
            print(f"[WEBHOOK] Signature: {signature}")
            return 'Forbidden', 403
            
        print("[WEBHOOK] Signature validated successfully")
        
    except Exception as e:
        print(f"[WEBHOOK] Validation error: {e}")
        import traceback
        traceback.print_exc()
    
    incoming_msg = request.values.get('Body', '').strip().lower()
    from_number = request.values.get('From', '')
    
    resp = MessagingResponse()
    msg = resp.message()
    
    inventory = load_inventory()
    
    if incoming_msg == 'alert':
        zero_items = [item for item, count in inventory.items() if count == 0]
        
        if not zero_items:
            msg.body("No items at zero quantity")
        else:
            if len(zero_items) == 1:
                response_text = f"🚨 PANTRY ALERT\n\nOut of stock:\n• {zero_items[0]}"
            else:
                items_list = "\n• ".join(zero_items)
                response_text = f"🚨 PANTRY ALERT\n\nOut of stock ({len(zero_items)} items):\n• {items_list}"
            msg.body(response_text)
        
        print(f"[WEBHOOK] Sent alert to {from_number}")
    
    elif incoming_msg == 'list':
        if not inventory:
            msg.body("Inventory is empty. Scan items first.")
        else:
            response_text = f"📦 PANTRY INVENTORY\n\nTotal: {len(inventory)} items\n\n"
            
            for item, count in sorted(inventory.items()):
                if count == 0:
                    response_text += f"❌ {item}: OUT OF STOCK\n"
                else:
                    response_text += f"✓ {item}: {count}\n"
            
            msg.body(response_text)
        
        print(f"[WEBHOOK] Sent full list to {from_number}")
    
    else:
        msg.body("Commands:\n'alert' - Get zero items\n'list' - Get full inventory")
        print(f"[WEBHOOK] Unknown command: {incoming_msg}")
    
    response = str(resp)
    print(f"[WEBHOOK] Sending TwiML response")
    return response, 200, {'Content-Type': 'text/xml'}


def main():
    print("\nSmart Pantry Inventory System")
    print("AI-Powered Item Detection & Tracking\n")
    
    print("Initializing system...")
    try:
        client = get_bedrock_client()
        print("[OK] AWS Bedrock connected")
    except Exception as e:
        print(f"[ERROR] Bedrock initialization failed: {e}")
        exit(1)
    
    try:
        with open('twilio_config.json', 'r') as f:
            json.load(f)
        print("[OK] WhatsApp configured")
    except FileNotFoundError:
        print("[WARN] WhatsApp not configured")
    
    print(f"[OK] Inventory file: {INVENTORY_FILE}")
    
    print("\nCAMERA SELECTION")
    print("0 - Built-in camera (default)")
    print("1 - External USB camera")
    print("2 - Other")
    
    camera_input = input("Select camera [0-2] (default 0): ").strip()
    camera_index = int(camera_input) if camera_input.isdigit() else 0
    
    print(f"\nSelected camera: {camera_index}")
    
    test_frame = capture_snapshot(camera_index, countdown=0)
    if test_frame is None:
        print(f"[ERROR] Camera {camera_index} not available")
        print("Try a different camera number")
        exit(1)
    print("[OK] Camera operational")
    
    print("\nCAPTURE DELAY")
    print("Countdown before capture (for positioning)")
    
    delay_input = input("Delay in seconds [0-10] (default 3): ").strip()
    countdown = int(delay_input) if delay_input.isdigit() and 0 <= int(delay_input) <= 10 else 3
    
    print(f"\nCapture delay: {countdown} seconds")
    
    print("\nDETECTION MODE")
    print("1 - General (detect any objects)")
    print("2 - Pantry (food/drink items only)")
    
    mode_choice = input("Select mode [1-2] (default 1): ").strip()
    mode = 'pantry' if mode_choice == '2' else 'general'
    
    print(f"\nMode: {mode.upper()}")
    
    print("\nCOMMANDS:")
    print("ENTER  - Take scan")
    print("show   - Display current inventory")
    print("server - Start WhatsApp webhook server")
    print("reset  - Clear inventory")
    print("quit   - Exit")
    
    while True:
        print("\n")
        user_input = input("Command: ").strip().lower()
        
        if user_input == 'quit':
            print("\nSystem shutting down...\n")
            break
        
        if user_input == 'reset':
            save_inventory({})
            print("\nInventory cleared.\n")
            continue
        
        if user_input == 'show':
            show_inventory()
            continue
        
        if user_input == 'server':
            print("\nStarting WhatsApp webhook server...")
            print("Server will run on http://localhost:8080/whatsapp")
            print("\nTo receive WhatsApp commands:")
            print("1. Install ngrok: brew install ngrok (Mac) or download from ngrok.com")
            print("2. Run: ngrok http 8080")
            print("3. Copy the https URL (e.g., https://abc123.ngrok.io)")
            print("4. Go to Twilio Console > WhatsApp Sandbox Settings")
            print("5. Set 'When a message comes in' to: https://abc123.ngrok.io/whatsapp")
            print("\nNow you can send 'alert' or 'list' to your WhatsApp number!")
            print("\nPress Ctrl+C to stop server\n")
            
            try:
                app.run(host='0.0.0.0', port=8080, debug=False)
            except KeyboardInterrupt:
                print("\n\nServer stopped\n")
            continue
        
        if user_input == '' or user_input == 'scan':
            process_scan(client, mode, camera_index, countdown)
        else:
            print("Unknown command. Try: scan, show, server, reset, quit")
    
    print("System stopped.\n")


if __name__ == "__main__":
    main()
