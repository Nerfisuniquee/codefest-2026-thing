# Smart Pantry

# Smart Pantry + WhatsApp Assist (POC)

This is a scrappy POC that uses a camera to scan pantry items and WhatsApp to control everything. It also has an assist mode that tries to guide a hand toward a target item using spoken directions.

## What it can do
- Scan a shelf and keep a simple inventory (AI vision via AWS Bedrock)
- Reply to WhatsApp commands (`list`, `alert`, `assist <item>`)
- Assist mode: track your hand + target item and say where to move

## You’ll need
- macOS (uses the built‑in `say` voice)
- Python 3.9+
- A camera connected to the Mac
- Twilio WhatsApp Sandbox (or a WhatsApp-enabled Twilio number)
- AWS Bedrock access

## Install deps
```bash
python3 -m pip install --upgrade pip
python3 -m pip install opencv-python mediapipe boto3 flask twilio
```

If MediaPipe fails to install, use this instead:
```bash
PYTHONIOENCODING=utf-8 python3 -m pip install --no-compile mediapipe
```

## Configuration
Create these files in the project folder:

`aws_config.json`
```json
{
  "region": "us-east-1",
  "aws_access_key_id": "YOUR_KEY",
  "aws_secret_access_key": "YOUR_SECRET"
}
```

`twilio_config.json`
```json
{
  "account_sid": "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "auth_token": "your_auth_token"
}
```

## Run
```bash
python3 /Users/krishnamjhunjhunwala/SmartPantry/smart_pantry_whatsapp.py
```

Use the console menu to select a camera, then start the server.

## WhatsApp Commands
- `alert` : list out-of-stock items
- `list` : full inventory
- `assist <item>` : start guidance mode for a target item
- `stop` : stop guidance

Example:
```
assist drink bottle
```

## Assist Mode Notes
- Uses the camera to detect the target item and your hand.
- If the item is briefly blocked, the last seen position is remembered for 6 seconds.
- Uses macOS `say` for speech output (set your Bluetooth speaker as system output).

## Twilio Validation
During local testing, signature validation can be disabled:

In `smart_pantry_whatsapp.py`:
```python
SKIP_TWILIO_VALIDATION = True
```

Set it back to `False` for production.

## Troubleshooting
If WhatsApp replies with 403:
- Your Twilio `auth_token` does not match the account that owns the WhatsApp sandbox.
- Make sure the webhook URL in Twilio matches the current ngrok URL.

If MediaPipe says `solutions` missing:
- Reinstall with `--no-compile` as shown above.

