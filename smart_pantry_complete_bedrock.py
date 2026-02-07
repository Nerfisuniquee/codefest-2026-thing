"""
Smart Pantry Inventory System
AI-powered item detection and tracking using AWS Bedrock Claude

Author: Smart Pantry Team
Date: 2026
Category: HealthTech & Digital Wellbeing
"""

import cv2
import boto3
import json
import base64
import time
import re
from datetime import datetime

INVENTORY_FILE = 'smart_pantry_inventory.json'


def load_aws_config():
    """Load AWS credentials from config file."""
    try:
        with open('aws_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_bedrock_client():
    """Initialize AWS Bedrock client."""
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


def send_whatsapp_alert(items_at_zero):
    """Send WhatsApp notification for items at zero quantity."""
    try:
        from twilio.rest import Client
        
        with open('twilio_config.json', 'r') as f:
            config = json.load(f)
        
        client = Client(config['account_sid'], config['auth_token'])
        
        if len(items_at_zero) == 1:
            message_body = f"🚨 *PANTRY ALERT*\n\nYou're out of: {items_at_zero[0]}\n\nAdd to shopping list?"
        else:
            items_list = "\n• ".join(items_at_zero)
            message_body = f"🚨 *PANTRY ALERT*\n\nYou're out of {len(items_at_zero)} items:\n• {items_list}\n\nAdd to shopping list?"
        
        message = client.messages.create(
            body=message_body,
            from_=config.get('twilio_whatsapp', 'whatsapp:+14155238886'),
            to=config.get('your_whatsapp', f"whatsapp:{config['your_phone']}")
        )
        
        print(f"WhatsApp sent. Message ID: {message.sid}")
        return True
        
    except FileNotFoundError:
        print("twilio_config.json not found - WhatsApp alerts disabled")
        return False
    except Exception as e:
        print(f"WhatsApp send failed: {e}")
        return False


def load_inventory():
    """Load inventory from file."""
    try:
        with open(INVENTORY_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_inventory(inventory):
    """Save inventory to file."""
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(inventory, f, indent=2)


def compare_inventory(old_inv, new_inv):
    """Compare two inventories and identify changes."""
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
        
        if old_count > 0 and new_count == 0:
            changes['zero_items'].append(item)
    
    return changes


def capture_snapshot(camera_index=0, countdown=3):
    """Capture image from webcam with countdown."""
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
    """Convert OpenCV image to base64 string."""
    success, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8') if success else None


def detect_items_with_bedrock(frame, client, mode='general'):
    """Detect items in image using AWS Bedrock Claude."""
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
    """Execute complete scan workflow."""
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
    new_inventory = detect_items_with_bedrock(frame, client, mode)
    
    if new_inventory is None:
        print("ERROR: Detection failed")
        return
    
    print(f"\nDetection complete:")
    for item, count in new_inventory.items():
        print(f"{item}: {count}")
    
    changes = compare_inventory(old_inventory, new_inventory)
    
    if changes['added']:
        print("\nItems added:")
        for item, count in changes['added'].items():
            print(f"+{count} {item}")
    
    if changes['removed']:
        print("\nItems removed:")
        for item, count in changes['removed'].items():
            print(f"-{count} {item}")
    
    if changes['zero_items']:
        print("\nALERT - Items at zero quantity:")
        for item in changes['zero_items']:
            print(f"{item}")
        
        print("\nSending WhatsApp notification...")
        if send_whatsapp_alert(changes['zero_items']):
            print("Notification sent successfully")
        else:
            print("Notification not configured")
    
    save_inventory(new_inventory)
    print(f"\nInventory updated\n")


def main():
    """Main program execution."""
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
        print("[OK] WhatsApp alerts configured")
    except FileNotFoundError:
        print("[WARN] WhatsApp alerts not configured")
    
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
    print("ENTER - Take scan")
    print("reset - Clear inventory")
    print("quit  - Exit")
    
    while True:
        print("\n")
        user_input = input("Press ENTER to scan (or 'reset'/'quit'): ").strip().lower()
        
        if user_input == 'quit':
            print("\nSystem shutting down...\n")
            break
        
        if user_input == 'reset':
            save_inventory({})
            print("\nInventory cleared.\n")
            continue
        
        process_scan(client, mode, camera_index, countdown)
    
    print("System stopped.\n")


if __name__ == "__main__":
    main()