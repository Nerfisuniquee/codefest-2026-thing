import cv2
import boto3
import json
import base64
import time
import re
import threading
import subprocess
from datetime import datetime
from flask import Flask, request

INVENTORY_FILE = 'smart_pantry_inventory.json'

app = Flask(__name__)
ASSIST_CLIENT = None
ASSIST_CAMERA_INDEX = 0
ASSIST_THREAD = None
ASSIST_STOP = threading.Event()
ASSIST_DEBUG = True  # Always show debug window during assist
SKIP_TWILIO_VALIDATION = True  # Set to False for production
DEBUG_SNAPSHOT_INTERVAL_SEC = 3


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


def detect_target_bbox_with_bedrock(frame, client, target_name):
    """
    Ask Bedrock to find a specific item and return its bounding box.
    Bounding box is normalized [0..1] as (x_min, y_min, x_max, y_max).
    """
    image_base64 = image_to_base64(frame)
    if not image_base64:
        return None

    prompt = f"""Find the item named "{target_name}" in this image.

Return ONLY this JSON:
{{
  "found": true/false,
  "bbox": [x_min, y_min, x_max, y_max],
  "confidence": 0-1
}}

Rules:
- If the item is not visible, set "found" to false and use bbox [0,0,0,0]
- Coordinates must be normalized between 0 and 1
- Use tight boxes around the item
"""

    try:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
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

        json_match = re.search(r'\{[\s\S]*"found"[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response_text

        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        result = json.loads(json_str.strip())
        if not result.get("found"):
            return None

        bbox = result.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            return None

        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            return None

        return {
            "bbox": bbox,
            "confidence": result.get("confidence", 0.0)
        }

    except Exception as e:
        print(f"Target detection error: {e}")
        return None




def speak(text):
    if not text:
        return
    try:
        subprocess.run(["say", text], check=False)
    except Exception as e:
        print(f"[SPEAK] Error: {e}")


def init_hands_detector():
    try:
        import mediapipe as mp
        return mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    except Exception as e:
        print("[HAND] MediaPipe not available. Install with: pip install mediapipe")
        print(f"[HAND] Error: {e}")
        return None


def get_hand_center(frame, hands_detector):
    if hands_detector is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    xs = [lm.x for lm in hand.landmark]
    ys = [lm.y for lm in hand.landmark]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def guidance_phrase(dx, dy):
    small = 0.06
    medium = 0.14

    x_phrase = None
    if dx > medium:
        x_phrase = "move right more"
    elif dx > small:
        x_phrase = "move right a bit"
    elif dx < -medium:
        x_phrase = "move left more"
    elif dx < -small:
        x_phrase = "move left a bit"

    y_phrase = None
    if dy > medium:
        y_phrase = "move down more"
    elif dy > small:
        y_phrase = "move down a bit"
    elif dy < -medium:
        y_phrase = "move up more"
    elif dy < -small:
        y_phrase = "move up a bit"

    parts = []
    if x_phrase:
        parts.append(x_phrase)
    if y_phrase:
        parts.append(y_phrase)

    if not parts:
        return "good position"
    return " and ".join(parts)


def draw_debug_overlay(frame, target_bbox, hand_center, target_name):
    h, w = frame.shape[:2]
    if target_bbox:
        x_min, y_min, x_max, y_max = target_bbox
        x1, y1 = int(x_min * w), int(y_min * h)
        x2, y2 = int(x_max * w), int(y_max * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, target_name, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    if hand_center:
        hx, hy = int(hand_center[0] * w), int(hand_center[1] * h)
        cv2.circle(frame, (hx, hy), 8, (0, 0, 255), -1)
        cv2.putText(frame, "hand", (hx + 10, hy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



def assist_loop(client, camera_index, target_name):
    print(f"[ASSIST] Starting guidance for: {target_name}")
    speak(f"Guidance started for {target_name}")

    cam = cv2.VideoCapture(camera_index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    hands = init_hands_detector()

    last_target_time = 0
    last_guidance_time = 0
    last_snapshot_time = 0
    target_center = None
    target_bbox = None

    while not ASSIST_STOP.is_set():
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.1)
            continue

        now = time.time()

        if now - last_target_time > 2.5:
            target = detect_target_bbox_with_bedrock(frame, client, target_name)
            if target:
                x_min, y_min, x_max, y_max = target["bbox"]
                target_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                target_bbox = (x_min, y_min, x_max, y_max)
            else:
                target_center = None
                target_bbox = None
            last_target_time = now

        hand_center = get_hand_center(frame, hands)

        if ASSIST_DEBUG and now - last_snapshot_time > DEBUG_SNAPSHOT_INTERVAL_SEC:
            debug_frame = frame.copy()
            draw_debug_overlay(debug_frame, target_bbox, hand_center, target_name)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = f"assist_debug_{ts}.jpg"
            cv2.imwrite(debug_file, debug_frame)
            print(f"[ASSIST] Saved debug snapshot: {debug_file}")
            last_snapshot_time = now

        if now - last_guidance_time > 1.0:
            if target_center and hand_center:
                dx = target_center[0] - hand_center[0]
                dy = target_center[1] - hand_center[1]
                phrase = guidance_phrase(dx, dy)
                print(f"[ASSIST] {phrase}")
                speak(phrase)
            elif not target_center:
                print("[ASSIST] I cannot see target")
                speak(f"I cannot see {target_name}")
            elif not hand_center:
                print("[ASSIST] I cannot see hand")
                speak("I cannot see your hand")
            last_guidance_time = now

        time.sleep(0.05)

    cam.release()
    print("[ASSIST] Stopped guidance")


def start_assist(client, camera_index, target_name):
    global ASSIST_THREAD
    stop_assist()
    ASSIST_STOP.clear()
    ASSIST_THREAD = threading.Thread(
        target=assist_loop,
        args=(client, camera_index, target_name),
        daemon=True,
    )
    ASSIST_THREAD.start()


def stop_assist():
    ASSIST_STOP.set()


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
    if not SKIP_TWILIO_VALIDATION:
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
    else:
        print("[WEBHOOK] Signature validation skipped (dev mode)")
    
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
                response_text = f"PANTRY ALERT\n\nOut of stock:\n- {zero_items[0]}"
            else:
                items_list = "\n- ".join(zero_items)
                response_text = f"PANTRY ALERT\n\nOut of stock ({len(zero_items)} items):\n- {items_list}"
            msg.body(response_text)
        
        print(f"[WEBHOOK] Sent alert to {from_number}")
    
    elif incoming_msg == 'list':
        if not inventory:
            msg.body("Inventory is empty. Scan items first.")
        else:
            response_text = f"PANTRY INVENTORY\n\nTotal: {len(inventory)} items\n\n"
            
            for item, count in sorted(inventory.items()):
                if count == 0:
                    response_text += f"- {item}: OUT OF STOCK\n"
                else:
                    response_text += f"- {item}: {count}\n"
            
            msg.body(response_text)
        
        print(f"[WEBHOOK] Sent full list to {from_number}")

    elif incoming_msg.startswith('assist '):
        target = incoming_msg.replace('assist', '', 1).strip()
        if not target:
            msg.body("Usage: assist <item>")
        else:
            if ASSIST_CLIENT is None:
                msg.body("Assist mode not ready. Start the server from the main app first.")
            else:
                start_assist(ASSIST_CLIENT, ASSIST_CAMERA_INDEX, target)
                msg.body(f"Assist started for: {target}\nSend 'stop' to end.")
        print(f"[WEBHOOK] Assist requested: {target}")

    elif incoming_msg == 'stop':
        stop_assist()
        msg.body("Assist stopped.")
        print(f"[WEBHOOK] Assist stopped by {from_number}")
    
    else:
        msg.body("Commands:\n'alert' - Get zero items\n'list' - Get full inventory\n'assist <item>' - Start guidance\n'stop' - Stop guidance")
        print(f"[WEBHOOK] Unknown command: {incoming_msg}")
    
    response = str(resp)
    print(f"[WEBHOOK] Sending TwiML response")
    return response, 200, {'Content-Type': 'text/xml'}


def main():
    global ASSIST_CLIENT, ASSIST_CAMERA_INDEX
    print("\nSmart Pantry Inventory System")
    print("AI-Powered Item Detection & Tracking\n")
    
    print("Initializing system...")
    try:
        client = get_bedrock_client()
        ASSIST_CLIENT = client
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
    ASSIST_CAMERA_INDEX = camera_index
    
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
            print("\nNow you can send 'alert', 'list', 'assist <item>', or 'stop' to WhatsApp!")
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
